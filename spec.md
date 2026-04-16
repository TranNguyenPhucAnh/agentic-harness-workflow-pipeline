# SPEC — IoT MLOps Landing Page
# Version: 1.0.0
# Last updated: 2025-04-16
# Agent: Claude (Spec Normalizer)
# Downstream consumers: Gemini 2.5 Flash (scaffold), Qwen 3.6 Plus, Deepseek v3.2 (implement)

---

## 0. Meta — pipeline instructions for downstream agents

```
SCAFFOLD_FORMAT : JSON_STRUCTURED
TEST_RUNNER     : vitest
FRAMEWORK       : React 18 + Vite 5 + TypeScript
DEPLOY_TARGET   : Cloudflare Pages
ITERATION_CAP   : 3
OUTPUT_DIRS     : src/, tests/
```

Gemini: output MUST be a single JSON object matching schema in §8.
Qwen / Deepseek: receive spec.md + scaffold JSON → implement ONLY the files listed in scaffold → do not add new files.

---

## 1. Project overview

**Product name:** AirGuard — IoT Air Quality Anomaly Detection  
**Purpose:** Portfolio landing page for a recruiter / tech lead audience.  
**Core message:** Show results, not process. Visualise what the system detected, how accurate the model is, and why it matters — without requiring the visitor to read code.

**Primary audience:**
- Recruiters (non-technical): want to see numbers, visual impact, clear outcome
- Tech leads (technical): want to see model quality, pipeline architecture hint, GitHub links

**Not in scope for this page:** live data, auth, backend, database. All data is generated client-side (demo mode).

---

## 2. Tech stack

| Layer | Choice |
|---|---|
| UI framework | React 18 + TypeScript |
| Build tool | Vite 5 |
| Styling | Tailwind CSS v3 |
| Charts | Recharts 2.x |
| Testing | Vitest + @testing-library/react |
| Deploy | Cloudflare Pages (static export) |
| State | React useState / useReducer only — no Redux |

---

## 3. Pages & routing

Single-page application. No router needed. All sections on one scrollable page.

**Sections in order:**
1. `<HeroSection>`
2. `<SummaryStickyBar>` (fixed, appears after hero scrolls out)
3. `<ResultCardsSection>`
4. `<ReplayDashboardSection>`
5. `<AnomalyBreakdownSection>`
6. `<ModelPerformanceSection>`
7. `<PipelineMiniSection>`
8. `<FooterSection>`

---

## 4. Component specifications

### 4.1 `SummaryStickyBar`

**File:** `src/components/SummaryStickyBar.tsx`

**Props:**
```ts
interface SummaryStickyBarProps {
  totalAnomalies: number;      // e.g. 47
  highestSeverity: 'LOW' | 'MED' | 'HIGH';
  modelAuc: number;            // e.g. 0.91
  precision: number;           // 0–1, e.g. 0.87
  monitoringDays: number;      // e.g. 7
}
```

**Behaviour:**
- Fixed top bar, z-index above all content
- Shows: monitoring window | anomaly count (red) | highest severity | AUC (green) | precision
- "REPLAY MODE" badge with animated pulse dot

**Tests required:**
- renders all 5 stat values
- anomaly count has class / colour indicating danger
- AUC has class / colour indicating success

---

### 4.2 `ReplayControls`

**File:** `src/components/ReplayControls.tsx`

**Props:**
```ts
interface ReplayControlsProps {
  totalPoints: number;
  currentIndex: number;
  isPlaying: boolean;
  speed: 1 | 5 | 20;
  onPlay: () => void;
  onPause: () => void;
  onReset: () => void;
  onSpeedChange: (s: 1 | 5 | 20) => void;
  onScrub: (index: number) => void;
  onJumpToNextAnomaly: () => void;
}
```

**Behaviour:**
- Play / Pause toggle button
- Reset button
- Speed selector: 1× / 5× / 20×  (active state highlighted)
- Scrub slider: range input 0 → totalPoints-1
- "Jump to anomaly" button: jumps currentIndex to next anomaly position after current
- Time label: "Day N, HH:MM" format derived from currentIndex

**Tests required:**
- clicking play calls onPlay
- clicking pause calls onPause
- clicking reset calls onReset
- speed button 20× calls onSpeedChange(20)
- scrub input change calls onScrub with numeric value
- "Jump to anomaly" button calls onJumpToNextAnomaly

---

### 4.3 `AnomalyFeed`

**File:** `src/components/AnomalyFeed.tsx`

**Props:**
```ts
interface AnomalyEvent {
  index: number;
  timestamp: Date;
  type: 'Gas spike' | 'Temp excursion' | 'Humidity anomaly';
  severity: 'LOW' | 'MED' | 'HIGH';
  decisionScore: number;   // negative float, e.g. -0.23
}

interface AnomalyFeedProps {
  events: AnomalyEvent[];
  maxVisible?: number;     // default 12
}
```

**Behaviour:**
- Renders a table/list of anomaly events, sorted newest first
- Each row: timestamp | type | decision score | severity badge
- Severity badge colours: HIGH=red, MED=amber, LOW=green
- Shows at most `maxVisible` rows

**Tests required:**
- renders correct number of rows (respects maxVisible)
- HIGH severity badge has correct danger colour class
- rows sorted newest-first (first row timestamp > last row timestamp)
- empty events array renders empty state message

---

### 4.4 `ModelGates`

**File:** `src/components/ModelGates.tsx`

**Props:**
```ts
interface GateResult {
  metric: 'ROC-AUC' | 'Precision' | 'Recall';
  value: number;      // 0–1
  threshold: number;  // 0–1
  passed: boolean;
}

interface ModelGatesProps {
  gates: GateResult[];
}
```

**Behaviour:**
- Renders each gate as a row: checkmark | metric name | progress bar | actual value | min threshold
- Progress bar width = value * 100%
- Passed gate: green checkmark + green bar
- Failed gate: red ✗ + red bar

**Tests required:**
- passed gate renders green indicator
- failed gate renders red indicator
- progress bar width matches value (within 1%)
- renders all 3 default gates when given standard input

---

### 4.5 `useSensorData` hook

**File:** `src/hooks/useSensorData.ts`

**Signature:**
```ts
interface SensorPoint {
  index: number;
  timestamp: Date;
  iaq: number;
  tempC: number;
  humidityPct: number;
  gasResistanceKohm: number;
  isAnomaly: boolean;
  anomalyMeta: AnomalyEvent | null;
  decisionScore: number;
}

interface UseSensorDataReturn {
  points: SensorPoint[];
  anomalyEvents: AnomalyEvent[];
  anomalyIndices: number[];
  stats: {
    totalPoints: number;
    totalAnomalies: number;
    anomalyRate: number;   // 0–1
    highestSeverity: 'LOW' | 'MED' | 'HIGH';
  };
}

function useSensorData(days?: number): UseSensorDataReturn
```

**Behaviour:**
- `days` defaults to 7
- Generates `days × 288` points (5-minute intervals)
- Anomalies injected at morning (07–09h) and evening (18–21h) clusters, plus random scatter
- Anomaly rate ≈ 0.4% (roughly 47/10080 for 7 days)
- Gas resistance drops sharply on Gas spike anomalies
- Temperature exceeds 33°C on Temp excursion anomalies
- Humidity exceeds 70% on Humidity anomaly events
- decisionScore: negative for anomalies (~-0.05 to -0.45), positive for normal (~0.02 to 0.15)

**Tests required:**
- returns exactly `days × 288` points for any `days` input
- anomalyIndices.length > 0
- all anomaly points have isAnomaly === true
- all normal points have isAnomaly === false  
- anomaly rate is between 0.2% and 1.5%
- Gas spike anomalies have gasResistanceKohm significantly lower than baseline
- stats.highestSeverity is one of LOW / MED / HIGH

---

### 4.6 `useReplay` hook

**File:** `src/hooks/useReplay.ts`

**Signature:**
```ts
interface UseReplayOptions {
  totalPoints: number;
  anomalyIndices: number[];
  windowSize?: number;    // default 576 (2 days visible)
}

interface UseReplayReturn {
  currentIndex: number;
  isPlaying: boolean;
  speed: 1 | 5 | 20;
  play: () => void;
  pause: () => void;
  reset: () => void;
  setSpeed: (s: 1 | 5 | 20) => void;
  scrubTo: (index: number) => void;
  jumpToNextAnomaly: () => void;
  windowStart: number;   // currentIndex clamped to valid range
}

function useReplay(options: UseReplayOptions): UseReplayReturn
```

**Behaviour:**
- `play()` starts requestAnimationFrame loop advancing currentIndex by `speed` per frame at 60fps
- `pause()` stops the loop
- `reset()` sets currentIndex = 0, stops loop
- `jumpToNextAnomaly()` finds next anomalyIndex > currentIndex, sets currentIndex to max(0, target - 50)
- `windowStart` = min(currentIndex, totalPoints - windowSize)
- Stops automatically when currentIndex >= totalPoints - windowSize

**Tests required:**
- initial state: currentIndex=0, isPlaying=false, speed=5
- reset() sets currentIndex to 0
- scrubTo(500) sets currentIndex to 500
- setSpeed(20) sets speed to 20
- jumpToNextAnomaly() sets currentIndex near next anomaly (not before current)

---

## 5. Data contracts

### AnomalyEvent (shared type)
```ts
// src/types/sensor.ts
export type Severity = 'LOW' | 'MED' | 'HIGH';
export type AnomalyType = 'Gas spike' | 'Temp excursion' | 'Humidity anomaly';

export interface AnomalyEvent {
  index: number;
  timestamp: Date;
  type: AnomalyType;
  severity: Severity;
  decisionScore: number;
}

export interface SensorPoint {
  index: number;
  timestamp: Date;
  iaq: number;
  tempC: number;
  humidityPct: number;
  gasResistanceKohm: number;
  isAnomaly: boolean;
  anomalyMeta: AnomalyEvent | null;
  decisionScore: number;
}

export interface GateResult {
  metric: 'ROC-AUC' | 'Precision' | 'Recall';
  value: number;
  threshold: number;
  passed: boolean;
}
```

---

## 6. Demo data constants

```ts
// src/data/demoConstants.ts
export const DEMO_DAYS = 7;
export const POINTS_PER_DAY = 288;          // 5-min intervals
export const WINDOW_SIZE = 576;             // 2 days visible
export const TARGET_ANOMALY_RATE = 0.004;   // 0.4%
export const TEMP_NORMAL_RANGE = [28, 33];  // °C
export const HUMIDITY_NORMAL_RANGE = [60, 70]; // %
export const GAS_BASELINE_KOHM = 145;
export const MODEL_GATES: GateResult[] = [
  { metric: 'ROC-AUC',   value: 0.91, threshold: 0.75, passed: true },
  { metric: 'Precision', value: 0.87, threshold: 0.60, passed: true },
  { metric: 'Recall',    value: 0.73, threshold: 0.50, passed: true },
];
```

---

## 7. File tree expected after scaffold

```
src/
  types/
    sensor.ts
  data/
    demoConstants.ts
  hooks/
    useSensorData.ts
    useReplay.ts
  components/
    SummaryStickyBar.tsx
    ReplayControls.tsx
    AnomalyFeed.tsx
    ModelGates.tsx
  App.tsx
  main.tsx

tests/
  hooks/
    useSensorData.test.ts
    useReplay.test.ts
  components/
    SummaryStickyBar.test.tsx
    ReplayControls.test.tsx
    AnomalyFeed.test.tsx
    ModelGates.test.tsx
```

---

## 8. Gemini output schema (MUST follow exactly)

```json
{
  "scaffold_version": "string — semver e.g. 1.0.0",
  "files": [
    {
      "file_path": "src/types/sensor.ts",
      "purpose": "one-line description",
      "code": "full file content as string",
      "is_test": false
    },
    {
      "file_path": "tests/hooks/useSensorData.test.ts",
      "purpose": "unit tests for useSensorData hook",
      "code": "full file content as string",
      "is_test": true,
      "test_runner": "vitest",
      "test_count": 7
    }
  ],
  "implementation_instructions": {
    "for_qwen": "string — specific hints for Qwen 3.6 Plus",
    "for_deepseek": "string — specific hints for Deepseek v3.2"
  },
  "vite_config_notes": "string",
  "tailwind_config_notes": "string"
}
```

**Rules for Gemini:**
- `code` field for non-test files: interfaces + function signatures + JSDoc only — NO implementation body (use `throw new Error('not implemented')` or return type stubs)
- `code` field for test files: complete, runnable vitest tests
- Every component must have a corresponding test file
- Every hook must have a corresponding test file
- Do not add files not listed in §7

---

## 9. Qwen / Deepseek implementation rules

- Receive: `spec.md` + full scaffold JSON from Gemini
- Task: implement ONLY the function bodies in non-test files
- Do NOT modify test files
- Do NOT add new files
- TypeScript strict mode — no `any`
- All Recharts usage: import from `recharts` directly
- Tailwind only — no inline styles, no CSS modules
- On ambiguity: follow the spec, not your own judgment

---

## 10. Acceptance criteria (all must pass for pipeline to succeed)

| # | Criterion |
|---|---|
| AC-1 | All vitest tests pass (exit code 0) |
| AC-2 | `vite build` completes with no errors |
| AC-3 | No TypeScript errors (`tsc --noEmit`) |
| AC-4 | `useSensorData(7).points.length === 2016` |
| AC-5 | `AnomalyFeed` renders empty state when events=[] |
| AC-6 | `ModelGates` colours passed=green, failed=red |
| AC-7 | `ReplayControls` speed buttons reflect active state |

---

## 11. Change log

| Version | Date | Change | Author |
|---|---|---|---|
| 1.0.0 | 2025-04-16 | Initial spec | Claude (Spec Agent) |
