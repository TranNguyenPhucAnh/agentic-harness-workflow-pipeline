"""
modules/drag_and_drop.py
========================

Shared interactive text/file intake layer for pipeline scripts.

Purpose
-------
Normalize user input from multiple UX paths into one text bundle:

    - CLI inline text
    - CLI file paths
    - piped stdin
    - interactive pasted text
    - drag-and-drop file paths in terminal
    - attachment-only input

This module is intentionally model-agnostic. It does NOT produce provider-native
file/image attachment blocks. It reads files through a caller-provided
`read_file_fn(Path) -> str` callback and composes everything into a single text
document.

Primary API
-----------
    gather_text_file_bundle(...)

Expected usage
--------------
    from modules.drag_and_drop import gather_text_file_bundle

    bundle = gather_text_file_bundle(
        cli_text=args.text,
        cli_files=args.input,
        read_file_fn=_read_input_file,
        prompt_title="Enter requirement",
        prompt_body="Describe the feature or drag-drop files.",
        attachment_prompt="Attach requirement files if any",
        default_attachment_only_prompt="Please analyze the attached files.",
        allow_interactive=True,
    )

    requirement_text = bundle.text
    source_metadata = bundle.sources
    attachment_only = bundle.attachment_only

Notes
-----
- This module does not call sys.exit().
- Caller should catch RuntimeError and decide how to present/fail.
- File reading/tracking is delegated to read_file_fn.
"""

from __future__ import annotations

import hashlib
import os
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


# ════════════════════════════════════════════════════════════════════════════
# Data structures
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class InputSource:
    """
    Metadata for one input source.

    `text` is kept in-memory for composition, but callers should avoid writing it
    directly to machine-readable reports unless intended.
    """
    kind: str                  # "text" | "file" | "stdin"
    label: str                 # user-friendly label
    text: str
    path: Path | None = None
    chars: int = 0
    sha256: str = ""


@dataclass
class TextFileBundle:
    """
    Normalized result of text/file gathering.
    """
    text: str
    sources: list[InputSource]
    attachment_only: bool = False

    def source_dicts(self, include_text: bool = False) -> list[dict[str, object]]:
        """
        Convert sources to JSON-safe dicts.

        By default, omits full text content.
        """
        out: list[dict[str, object]] = []
        for src in self.sources:
            item = asdict(src)
            if src.path is not None:
                item["path"] = str(src.path)
            if not include_text:
                item.pop("text", None)
            out.append(item)
        return out


ReadFileFn = Callable[[Path], str]


# ════════════════════════════════════════════════════════════════════════════
# Small helpers
# ════════════════════════════════════════════════════════════════════════════

def _sha256_short(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _is_tty() -> bool:
    try:
        return sys.stdin.isatty()
    except Exception:
        return False


def _strip_wrapping_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2:
        if (value[0] == value[-1]) and value[0] in {"'", '"'}:
            return value[1:-1]
    return value


def _expand_path(raw: str) -> Path:
    """
    Expand user path string.

    Handles:
      - surrounding quotes
      - ~
      - environment variables
    """
    cleaned = _strip_wrapping_quotes(raw.strip())
    cleaned = os.path.expandvars(cleaned)
    cleaned = os.path.expanduser(cleaned)
    return Path(cleaned)


def _looks_like_probable_path(token: str) -> bool:
    """
    Heuristic used only for friendlier warnings, not authoritative validation.
    """
    t = token.strip()
    if not t:
        return False
    if t.startswith(("/", "./", "../", "~")):
        return True
    if "\\" in t:
        return True
    if ":" in t and len(t) >= 2:
        # Windows drive path like C:\...
        return True
    suffix = Path(t).suffix
    return bool(suffix)


# ════════════════════════════════════════════════════════════════════════════
# Shell/drag-drop path parsing
# ════════════════════════════════════════════════════════════════════════════

def parse_shell_paths(raw: str) -> list[str]:
    """
    Parse a raw terminal string into path-like tokens.

    Supports common drag/drop forms:
      macOS/Linux:
          /Users/me/My\\ Docs/spec.md
          '/Users/me/My Docs/spec.md'
          "/Users/me/My Docs/spec.md"

      Windows:
          C:\\Users\\me\\Desktop\\spec.md
          "C:\\Users\\me\\Desktop\\My Spec.md"

    This function does not validate that tokens exist.
    """
    raw = (raw or "").strip()
    if not raw:
        return []

    # First try shlex. POSIX mode is correct for macOS/Linux drag-drop escaping.
    # On Windows, posix=False preserves backslashes better.
    try:
        tokens = shlex.split(raw, posix=(os.name != "nt"))
    except ValueError:
        # Fallback: split by lines, then whitespace.
        tokens = []

    if tokens:
        return [t.strip() for t in tokens if t.strip()]

    fallback: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        fallback.extend(part.strip() for part in line.split() if part.strip())

    return fallback


def normalize_and_validate_file_paths(
    raw_paths: Iterable[str | os.PathLike[str]],
    *,
    must_exist: bool = True,
    files_only: bool = True,
) -> tuple[list[Path], list[str]]:
    """
    Normalize path strings and validate them.

    Returns:
        (valid_paths, invalid_items)

    Does not raise.
    """
    valid: list[Path] = []
    invalid: list[str] = []

    seen: set[str] = set()

    for raw in raw_paths:
        raw_str = str(raw).strip()
        if not raw_str:
            continue

        path = _expand_path(raw_str)

        try:
            resolved = path.resolve()
        except Exception:
            resolved = path.absolute()

        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)

        if must_exist and not resolved.exists():
            invalid.append(raw_str)
            continue

        if files_only and resolved.exists() and not resolved.is_file():
            invalid.append(raw_str)
            continue

        valid.append(resolved)

    return valid, invalid


def detect_attachment_only_input(raw_text: str) -> list[Path] | None:
    """
    Detect whether raw text is purely a list of existing file paths.

    Returns:
        list[Path] if every parsed token is a valid file
        None otherwise

    Important:
    - Empty input is not attachment-only.
    - Mixed text + paths is not attachment-only.
    - All tokens must resolve to existing files.
    """
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return None

    tokens = parse_shell_paths(raw_text)
    if not tokens:
        return None

    valid, invalid = normalize_and_validate_file_paths(tokens)

    if invalid:
        return None

    if not valid:
        return None

    # Require a 1:1 relation to avoid treating weird shlex artifacts as paths.
    if len(valid) != len(tokens):
        return None

    return valid


# ════════════════════════════════════════════════════════════════════════════
# Interactive prompts
# ════════════════════════════════════════════════════════════════════════════

def _print_input_box(title: str, body: str) -> None:
    title = title.strip() or "Enter input"
    body = body.strip()

    print()
    print("╔" + "═" * 62 + "╗")
    print(f"║  {title[:58].ljust(58)}  ║")
    print("╠" + "═" * 62 + "╣")

    if body:
        for line in _wrap_for_box(body, width=58):
            print(f"║  {line.ljust(58)}  ║")

    print("║  Paste multiple lines if needed.                           ║")
    print("║  Or drag-drop one or more files into the terminal.          ║")
    print("║  Press Ctrl-D (Mac/Linux) or Ctrl-Z Enter (Windows) to end. ║")
    print("╚" + "═" * 62 + "╝")


def _wrap_for_box(text: str, width: int = 58) -> list[str]:
    words = text.split()
    if not words:
        return []

    lines: list[str] = []
    current = ""

    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines


def prompt_for_text_or_paths(
    *,
    prompt_title: str = "Enter input",
    prompt_body: str = "",
) -> str:
    """
    Prompt user for freeform text or drag/drop path list.

    Uses EOF to finish because requirement text can be multiline.
    """
    _print_input_box(prompt_title, prompt_body)
    print("▶ ", end="", flush=True)

    try:
        return sys.stdin.read().strip()
    except KeyboardInterrupt:
        print()
        raise RuntimeError("Input aborted by user.")


def prompt_for_attachments(
    prompt: str = "Attach files if any",
    *,
    allow_retry: bool = True,
) -> list[Path]:
    """
    Prompt user for optional file attachments.

    User may:
      - press Enter to skip
      - drag-drop one or multiple files
      - paste shell-quoted paths

    Returns only valid file paths. Invalid entries are warned and ignored,
    unless all entries are invalid and allow_retry=True, in which case user is
    prompted again.
    """
    if not _is_tty():
        return []

    while True:
        print()
        print(f"[input] {prompt}")
        print("        Drag-drop file(s), paste path(s), or press Enter to skip.")
        raw = input("        → ").strip()

        if not raw:
            return []

        tokens = parse_shell_paths(raw)
        valid, invalid = normalize_and_validate_file_paths(tokens)

        if invalid:
            print("[input][warn] Some paths were not valid files:")
            for item in invalid:
                print(f"  - {item}")

        if valid:
            print(f"[input] Attached {len(valid)} file(s).")
            return valid

        if not allow_retry:
            return []

        print("[input] No valid files found. Try again or press Enter to skip.")


# ════════════════════════════════════════════════════════════════════════════
# Source creation and composition
# ════════════════════════════════════════════════════════════════════════════

def _make_text_source(kind: str, label: str, text: str, path: Path | None = None) -> InputSource:
    text = text or ""
    return InputSource(
        kind=kind,
        label=label,
        path=path,
        text=text,
        chars=len(text),
        sha256=_sha256_short(text),
    )


def read_file_sources(
    file_paths: Sequence[Path],
    *,
    read_file_fn: ReadFileFn,
    empty_file_policy: str = "warn",
) -> list[InputSource]:
    """
    Read files through caller-provided read_file_fn.

    empty_file_policy:
      - "warn": include empty source but print warning
      - "skip": skip empty source
      - "error": raise RuntimeError on empty text
    """
    sources: list[InputSource] = []

    for path in file_paths:
        try:
            text = read_file_fn(path)
        except Exception as exc:
            raise RuntimeError(f"Failed to read file {path}: {exc}") from exc

        if not text.strip():
            msg = f"File produced empty text: {path}"
            if empty_file_policy == "error":
                raise RuntimeError(msg)
            if empty_file_policy == "skip":
                print(f"[input][warn] {msg}; skipping.")
                continue
            print(f"[input][warn] {msg}")

        sources.append(
            _make_text_source(
                kind="file",
                label=path.name,
                text=text,
                path=path,
            )
        )

    return sources


def compose_text_from_sources(
    *,
    sources: Sequence[InputSource],
    default_attachment_only_prompt: str = "",
    attachment_only: bool = False,
) -> str:
    """
    Compose all sources into a single markdown-ish document.

    This keeps downstream agents simple: they only need one string.
    """
    parts: list[str] = []

    if attachment_only and default_attachment_only_prompt.strip():
        parts.extend(
            [
                "# Input Instruction",
                "",
                default_attachment_only_prompt.strip(),
                "",
            ]
        )

    text_sources = [s for s in sources if s.kind in {"text", "stdin"}]
    file_sources = [s for s in sources if s.kind == "file"]
    other_sources = [s for s in sources if s.kind not in {"text", "stdin", "file"}]

    if text_sources:
        if len(text_sources) == 1:
            src = text_sources[0]
            title = "Piped Requirement" if src.kind == "stdin" else "Inline Requirement"
            parts.extend(
                [
                    f"# {title}",
                    "",
                    src.text.strip(),
                    "",
                ]
            )
        else:
            parts.extend(["# Text Requirement Sources", ""])
            for idx, src in enumerate(text_sources, 1):
                parts.extend(
                    [
                        f"## Text Source {idx}: {src.label}",
                        "",
                        src.text.strip(),
                        "",
                    ]
                )

    if file_sources:
        parts.extend(["# Attached Requirement Sources", ""])

        for idx, src in enumerate(file_sources, 1):
            path_str = str(src.path) if src.path else src.label
            parts.extend(
                [
                    f"## File {idx}: {src.label}",
                    "",
                    f"Source path: `{path_str}`",
                    "",
                    src.text.strip(),
                    "",
                ]
            )

    if other_sources:
        parts.extend(["# Other Input Sources", ""])
        for idx, src in enumerate(other_sources, 1):
            parts.extend(
                [
                    f"## Source {idx}: {src.label}",
                    "",
                    src.text.strip(),
                    "",
                ]
            )

    return "\n".join(parts).strip()


# ════════════════════════════════════════════════════════════════════════════
# Main public API
# ════════════════════════════════════════════════════════════════════════════

def gather_text_file_bundle(
    *,
    cli_text: str | None = None,
    cli_files: Sequence[str | os.PathLike[str]] | None = None,
    read_file_fn: ReadFileFn,
    prompt_title: str = "Enter input",
    prompt_body: str = "",
    attachment_prompt: str = "Attach files if any",
    default_attachment_only_prompt: str = "Please analyze the attached files.",
    allow_interactive: bool = True,
    ask_for_attachments_after_text: bool = True,
) -> TextFileBundle:
    """
    Gather text/file input and normalize into TextFileBundle.

    Resolution order:
      1. CLI files
      2. CLI text
      3. piped stdin if stdin is not TTY
      4. interactive text/path prompt if allowed and TTY
      5. optional attachment prompt if text was provided and no attachment-only
      6. error if still empty

    Attachment-only behavior:
      - If cli_text/piped/interactive raw input consists only of valid file paths,
        those files are used as sources and text is replaced by the composed
        attachment document.
      - When attachment-only is detected, the extra attachment prompt is skipped.

    Raises:
      RuntimeError for invalid CLI files, no input, or read failures.
    """
    cli_files = list(cli_files or [])
    cli_text = (cli_text or "").strip()

    sources: list[InputSource] = []
    attachment_only = False
    already_prompted_primary_input = False

    # ── 1. CLI files ────────────────────────────────────────────────────────
    if cli_files:
        valid_files, invalid_files = normalize_and_validate_file_paths(cli_files)

        if invalid_files:
            details = "\n".join(f"  - {item}" for item in invalid_files)
            raise RuntimeError(f"Invalid --input file path(s):\n{details}")

        sources.extend(read_file_sources(valid_files, read_file_fn=read_file_fn))

    # ── 2. CLI text, with attachment-only detection ─────────────────────────
    if cli_text:
        detected = detect_attachment_only_input(cli_text)

        if detected:
            attachment_only = True
            sources.extend(read_file_sources(detected, read_file_fn=read_file_fn))
        else:
            sources.append(
                _make_text_source(
                    kind="text",
                    label="inline text",
                    text=cli_text,
                )
            )

    # ── 3. Piped stdin if no CLI text was provided ──────────────────────────
    if not cli_text and not cli_files and not _is_tty():
        piped = sys.stdin.read().strip()

        if piped:
            detected = detect_attachment_only_input(piped)

            if detected:
                attachment_only = True
                sources.extend(read_file_sources(detected, read_file_fn=read_file_fn))
            else:
                sources.append(
                    _make_text_source(
                        kind="stdin",
                        label="piped stdin",
                        text=piped,
                    )
                )

    # ── 4. Interactive primary input if still empty ─────────────────────────
    if not sources and allow_interactive and _is_tty():
        raw = prompt_for_text_or_paths(
            prompt_title=prompt_title,
            prompt_body=prompt_body,
        )
        already_prompted_primary_input = True

        if raw:
            detected = detect_attachment_only_input(raw)

            if detected:
                attachment_only = True
                sources.extend(read_file_sources(detected, read_file_fn=read_file_fn))
            else:
                sources.append(
                    _make_text_source(
                        kind="text",
                        label="interactive text",
                        text=raw,
                    )
                )

    # ── 5. Optional extra attachment prompt ─────────────────────────────────
    #
    # If primary input was attachment-only, skip this prompt.
    #
    # If user supplied CLI files already, no need to ask again by default.
    #
    # If user provided real text interactively, ask for optional extra files.
    # For CLI --text, also ask when interactive TTY, because long-term design is
    # user-friendly and less flag-oriented.
    if (
        ask_for_attachments_after_text
        and allow_interactive
        and _is_tty()
        and not attachment_only
        and sources
    ):
        has_file_source = any(src.kind == "file" for src in sources)

        # Avoid asking twice when user already explicitly provided --input.
        # But if only text was provided, ask for optional attachments.
        if not has_file_source:
            attached = prompt_for_attachments(attachment_prompt)
            if attached:
                sources.extend(read_file_sources(attached, read_file_fn=read_file_fn))

    # ── 6. Final validation ─────────────────────────────────────────────────
    if not sources:
        if allow_interactive and _is_tty() and already_prompted_primary_input:
            raise RuntimeError("No input provided.")
        raise RuntimeError(
            "No input provided. Use --input, --text, pipe stdin, "
            "or run interactively and paste text / drag-drop files."
        )

    text = compose_text_from_sources(
        sources=sources,
        default_attachment_only_prompt=default_attachment_only_prompt,
        attachment_only=attachment_only,
    )

    if not text.strip():
        raise RuntimeError("Input sources were collected but produced empty text.")

    return TextFileBundle(
        text=text,
        sources=sources,
        attachment_only=attachment_only,
    )


# ════════════════════════════════════════════════════════════════════════════
# Convenience API for callers that want only attachment parsing
# ════════════════════════════════════════════════════════════════════════════

def maybe_parse_dragged_files(raw: str) -> list[Path]:
    """
    Parse raw drag/drop string and return valid file paths.

    Unlike detect_attachment_only_input(), this returns valid files even if some
    tokens are invalid. Useful for optional attachment prompts.
    """
    tokens = parse_shell_paths(raw)
    valid, _invalid = normalize_and_validate_file_paths(tokens)
    return valid
