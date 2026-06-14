from anthropic import Anthropic
import argparse
import base64
import json
import mimetypes
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

#MODEL = os.environ.get("CKEY_MODEL", "gpt-5.5")
MODEL = os.environ.get("CKEY_MODEL", "claude-opus-4-7")
MAX_TOKENS = int(os.environ.get("CKEY_MAX_TOKENS", "2048"))
OUTPUT_DIR = os.environ.get("CKEY_OUTPUT_DIR", "outputs")
PREVIEW_CHARS = int(os.environ.get("CKEY_PREVIEW_CHARS", "800"))

# Giới hạn an toàn để tránh gửi file quá lớn
MAX_TEXT_FILE_BYTES = int(os.environ.get("CKEY_MAX_TEXT_FILE_BYTES", "200000"))
MAX_BINARY_FILE_BYTES = int(os.environ.get("CKEY_MAX_BINARY_FILE_BYTES", str(10 * 1024 * 1024)))

TEXT_EXTENSIONS = {
    ".py", ".txt", ".md", ".json", ".yaml", ".yml",
    ".js", ".ts", ".tsx", ".jsx", ".html", ".css",
    ".sh", ".toml", ".ini", ".csv", ".xml", ".log",
    ".env", ".sql", ".conf", ".rst", ".cfg",
}

MULTILINE_SENTINEL = "/done"
DEFAULT_ATTACHMENT_ONLY_PROMPT = "Hãy xem các file đính kèm."


def get_client() -> Anthropic:
    api_key = os.environ.get("CKEY_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Thiếu biến môi trường CKEY_API_KEY hoặc ANTHROPIC_API_KEY")

    return Anthropic(
        api_key=api_key,
        base_url="https://ckey.vn",
    )


def ensure_supported_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    if not path.is_file():
        raise ValueError(f"Không phải file hợp lệ: {path}")


def read_text_file(path: Path) -> str:
    file_size = path.stat().st_size
    if file_size > MAX_TEXT_FILE_BYTES:
        raise ValueError(
            f"File text quá lớn: {path.name} ({file_size} bytes). "
            f"Giới hạn hiện tại là {MAX_TEXT_FILE_BYTES} bytes."
        )

    return path.read_text(encoding="utf-8", errors="replace")


def read_binary_file_as_base64(path: Path) -> str:
    file_size = path.stat().st_size
    if file_size > MAX_BINARY_FILE_BYTES:
        raise ValueError(
            f"File binary quá lớn: {path.name} ({file_size} bytes). "
            f"Giới hạn hiện tại là {MAX_BINARY_FILE_BYTES} bytes."
        )

    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def file_to_block(file_path: str) -> dict[str, Any]:
    path = Path(file_path).expanduser().resolve()
    ensure_supported_file(path)

    ext = path.suffix.lower()
    mime_type, _ = mimetypes.guess_type(path.name)

    if ext in TEXT_EXTENSIONS or (mime_type and mime_type.startswith("text/")):
        text = read_text_file(path)
        return {
            "type": "text",
            "text": f"File: {path.name}\n\n{text}",
        }

    if mime_type and mime_type.startswith("image/"):
        encoded = read_binary_file_as_base64(path)
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": encoded,
            },
        }

    if mime_type == "application/pdf":
        encoded = read_binary_file_as_base64(path)
        return {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": encoded,
            },
        }

    raise ValueError(
        f"Chưa hỗ trợ loại file này: {path.name} ({mime_type or 'unknown'}). "
        "Hiện chỉ hỗ trợ text, image, pdf."
    )


def extract_text_from_anthropic_style(data: dict[str, Any]) -> str | None:
    content = data.get("content")
    if not isinstance(content, list):
        return None

    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text)

    if parts:
        return "\n".join(parts)

    return None


def extract_text_from_openai_style(data: dict[str, Any]) -> str | None:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None

    first = choices[0] or {}

    if isinstance(first, dict):
        message = first.get("message")
        if isinstance(message, dict):
            content = message.get("content")

            if isinstance(content, str) and content.strip():
                return content

            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") in {"text", "output_text"}:
                        text = item.get("text")
                        if isinstance(text, str) and text.strip():
                            parts.append(text)
                if parts:
                    return "\n".join(parts)

        text = first.get("text")
        if isinstance(text, str) and text.strip():
            return text

    return None


def extract_text_from_response_body(body_text: str) -> str:
    try:
        data = json.loads(body_text)
    except json.JSONDecodeError as e:
        return f"Không parse được JSON: {e}\nRaw: {body_text[:1000]}"

    anthropic_text = extract_text_from_anthropic_style(data)
    if anthropic_text:
        return anthropic_text

    openai_text = extract_text_from_openai_style(data)
    if openai_text:
        return openai_text

    return f"Không tìm thấy text trong response.\nRaw: {body_text[:1000]}"


def save_result_to_markdown(
    result: str,
    output_dir: str = OUTPUT_DIR,
    prefix: str = "ckey-response",
) -> Path:
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    out_file = out_dir / f"{prefix}-{timestamp}.md"

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(result)

    return out_file


def open_file_in_default_app(path: Path) -> None:
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        elif os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception as e:
        print(f"Không thể tự mở file: {e}")


def build_user_message(prompt: str, file_paths: list[str] | None = None) -> dict[str, Any]:
    prompt = prompt.strip()
    file_paths = file_paths or []

    if not prompt and not file_paths:
        raise ValueError("Prompt rỗng và không có file đính kèm.")

    blocks: list[dict[str, Any]] = []

    if prompt:
        blocks.append(
            {
                "type": "text",
                "text": prompt,
            }
        )
    elif file_paths:
        blocks.append(
            {
                "type": "text",
                "text": DEFAULT_ATTACHMENT_ONLY_PROMPT,
            }
        )

    for file_path in file_paths:
        cleaned = file_path.strip()
        if cleaned:
            blocks.append(file_to_block(cleaned))

    return {
        "role": "user",
        "content": blocks,
    }


def build_assistant_message(text: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": text,
            }
        ],
    }


def ask_with_history(history: list[dict[str, Any]]) -> str:
    client = get_client()

    raw_response = client.messages.with_raw_response.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=history,
    )

    body_text = raw_response.text
    return extract_text_from_response_body(body_text)


def print_preview(title: str, text: str) -> None:
    print(f"\n===== {title} =====\n")
    print(text[:PREVIEW_CHARS] + ("\n...\n" if len(text) > PREVIEW_CHARS else ""))


def handle_and_save_result(result: str, prefix: str, auto_open: bool) -> None:
    print_preview("KẾT QUẢ" if prefix == "ckey-response" else "ASSISTANT", result)

    out_file = save_result_to_markdown(result, prefix=prefix)
    print(f"\nĐã lưu response vào: {out_file}")

    if auto_open:
        open_file_in_default_app(out_file)


def parse_shell_paths(raw_paths: str) -> list[str]:
    if not raw_paths.strip():
        return []
    return shlex.split(raw_paths)


def normalize_and_validate_file_paths(paths: list[str]) -> tuple[list[str], list[str]]:
    valid_paths: list[str] = []
    invalid_paths: list[str] = []

    for raw in paths:
        cleaned = raw.strip()
        if not cleaned:
            continue

        resolved = Path(cleaned).expanduser()
        if resolved.exists() and resolved.is_file():
            valid_paths.append(str(resolved.resolve()))
        else:
            invalid_paths.append(raw)

    return valid_paths, invalid_paths


def detect_attachment_only_input(raw_text: str) -> list[str] | None:
    """
    Nếu raw_text thực chất là danh sách path hợp lệ được kéo-thả/paste vào terminal,
    trả về list file paths. Nếu không chắc chắn, trả về None để coi như prompt text bình thường.
    """
    stripped = raw_text.strip()
    if not stripped:
        return None

    try:
        parsed_paths = parse_shell_paths(stripped)
    except ValueError:
        return None

    if not parsed_paths:
        return None

    valid_paths, invalid_paths = normalize_and_validate_file_paths(parsed_paths)

    if valid_paths and not invalid_paths:
        return valid_paths

    return None


def prompt_for_attachments(
    prompt_text: str = "Files đính kèm (kéo-thả file vào đây, Enter để bỏ qua): "
) -> list[str]:
    while True:
        try:
            raw_paths = input(prompt_text).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBỏ qua attachments.")
            return []

        if not raw_paths:
            return []

        try:
            parsed_paths = parse_shell_paths(raw_paths)
        except ValueError as e:
            print(f"Không parse được danh sách file: {e}")
            print("Hãy thử kéo-thả lại file vào terminal, hoặc nhập path đã quote/escape đúng.")
            continue

        valid_paths, invalid_paths = normalize_and_validate_file_paths(parsed_paths)

        if valid_paths and not invalid_paths:
            return valid_paths

        if not valid_paths and invalid_paths:
            print("Input này không giống file path hợp lệ. Bỏ qua attachments.")
            return []

        print("Một số path không hợp lệ:")
        for p in invalid_paths:
            print(f" - {p}")
        print("Hãy thử lại, hoặc Enter để bỏ qua.")


def prompt_multiline_text(
    intro: str,
    done_token: str = MULTILINE_SENTINEL,
    allow_empty_first_line_exit: bool = False,
) -> str:
    print(intro)
    lines: list[str] = []

    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if allow_empty_first_line_exit and not lines and not line.strip():
            return ""

        if line.strip() == done_token:
            break

        lines.append(line)

    return "\n".join(lines).strip()


def prompt_initial_message() -> str:
    return prompt_multiline_text(
        intro=(
            "Nhập prompt đầu tiên (có thể paste nhiều dòng).\n"
            f"Kết thúc bằng dòng chỉ chứa {MULTILINE_SENTINEL}\n"
        ),
        done_token=MULTILINE_SENTINEL,
        allow_empty_first_line_exit=False,
    )


def prompt_chat_message() -> str:
    return prompt_multiline_text(
        intro=(
            f"\nBạn (paste nhiều dòng được, kết thúc bằng {MULTILINE_SENTINEL}; "
            "Enter rỗng ngay dòng đầu để thoát):"
        ),
        done_token=MULTILINE_SENTINEL,
        allow_empty_first_line_exit=True,
    )


def run_one_shot(prompt: str, file_paths: list[str], auto_open: bool) -> None:
    try:
        history = [build_user_message(prompt, file_paths)]
        result = ask_with_history(history)
    except Exception as e:
        print(f"Lỗi khi xử lý request: {e}")
        return

    handle_and_save_result(result, prefix="ckey-response", auto_open=auto_open)


def run_chat(initial_prompt: str | None, initial_files: list[str], auto_open: bool) -> None:
    history: list[dict[str, Any]] = []

    if initial_prompt or initial_files:
        try:
            user_message = build_user_message(initial_prompt or "", initial_files)
            history.append(user_message)

            reply = ask_with_history(history)
            history.append(build_assistant_message(reply))
        except Exception as e:
            print(f"Lỗi khi gửi prompt đầu tiên: {e}")
            return

        handle_and_save_result(reply, prefix="ckey-chat", auto_open=auto_open)

    while True:
        raw_input_text = prompt_chat_message()

        if not raw_input_text:
            print("Thoát chat.")
            break

        inline_files = detect_attachment_only_input(raw_input_text)

        if inline_files is not None:
            prompt = ""
            file_paths = inline_files
            print("Đã nhận diện file được kéo-thả trực tiếp trong ô chat. Bỏ qua bước hỏi attachments riêng.")
        else:
            prompt = raw_input_text
            file_paths = prompt_for_attachments(
                "Files đính kèm lượt này (kéo-thả file vào đây, Enter để bỏ qua): "
            )

        try:
            user_message = build_user_message(prompt, file_paths)
            history.append(user_message)

            reply = ask_with_history(history)
            history.append(build_assistant_message(reply))
        except Exception as e:
            print(f"Lỗi khi xử lý lượt chat: {e}")
            if history and history[-1].get("role") == "user":
                history.pop()
            continue

        handle_and_save_result(reply, prefix="ckey-chat", auto_open=auto_open)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI chat với Claude qua ckey.vn, hỗ trợ file attachments và multi-turn chat."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help='Prompt đầu tiên, ví dụ: python ckey.py "review file này"',
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="Danh sách file local. Có thể kéo-thả nhiều file sau --files",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Đọc prompt đầu tiên từ stdin.",
    )

    parser.add_argument(
        "--chat",
        dest="chat",
        action="store_true",
        help="Bật multi-turn chat mode (mặc định đang bật).",
    )
    parser.add_argument(
        "--no-chat",
        dest="chat",
        action="store_false",
        help="Tắt chat mode, chỉ chạy one-shot.",
    )

    parser.add_argument(
        "--open",
        dest="auto_open",
        action="store_true",
        help="Tự mở file output sau mỗi response (mặc định đang bật).",
    )
    parser.add_argument(
        "--no-open",
        dest="auto_open",
        action="store_false",
        help="Không tự mở file output.",
    )

    parser.set_defaults(chat=True, auto_open=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    initial_prompt = args.prompt
    initial_files = args.files

    if args.stdin:
        initial_prompt = sys.stdin.read().strip()
        if not initial_prompt:
            raise SystemExit("Không đọc được prompt từ stdin, kết thúc.")
    elif not initial_prompt:
        try:
            initial_prompt = prompt_initial_message()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("\nKết thúc.")

        if not initial_prompt:
            raise SystemExit("Không có prompt, kết thúc.")

    inline_initial_files = detect_attachment_only_input(initial_prompt)
    if inline_initial_files is not None:
        initial_files = inline_initial_files
        initial_prompt = ""
        print("Đã nhận diện file được kéo-thả trực tiếp trong prompt đầu tiên. Bỏ qua bước hỏi attachments riêng.")

    # Fail-safe: nếu user chưa truyền --files cho prompt đầu tiên
    # thì hỏi riêng một prompt attachments để kéo-thả file.
    if initial_files is None:
        initial_files = prompt_for_attachments(
            "Files đính kèm cho prompt đầu tiên (kéo-thả file vào đây, Enter để bỏ qua): "
        )

    if args.chat:
        run_chat(initial_prompt, initial_files, args.auto_open)
        return

    run_one_shot(initial_prompt, initial_files, args.auto_open)


if __name__ == "__main__":
    main()

