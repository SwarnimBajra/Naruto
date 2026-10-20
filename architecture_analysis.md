# Hand Seal Dojo — Architecture & Evidence Analysis

---

## 1. What a Top Practitioner Would Do Beyond What You Asked (and What They'd Skip)

### They would add

| Area | What | Why |
|---|---|---|
| **Record raw frames, not just events** | Save the full per-frame score vector + landmark vector + timestamp for every session, not just the `new_confirm` boolean that [`service.py`](file:///d:/hand-seal-dojo/perception/service.py#L252) emits | You cannot tune the [StabilityGate](file:///d:/hand-seal-dojo/perception/service.py#L60-L82) offline without the raw time series. Your gate params (`WINDOW=12`, `NEED=8`, `ARM_BELOW=0.35`, `ARM_FRAMES=4`) were found by watching yourself; a replay harness would let you sweep them against labelled sessions from other people. |
| **Collect from diverse strangers early** | 5–10 consented volunteers, each with a different webcam and lighting | Your spike data ([`spike_yolox.csv`](file:///d:/hand-seal-dojo/data/spike_yolox.csv), [`spike_mediapipe_baseline.csv`](file:///d:/hand-seal-dojo/data/spike_mediapipe_baseline.csv)) is one person, one room. MediaPipe detects **0 % two-hands** on several seals like Tiger, Dragon, Hare, Ram — this may or may not generalise. |
| **Define the correctness ground truth from primary sources** | Naruto manga panels or Databook diagrams for each of the 12 seals; two independent raters annotating whether a player's pose matches | Right now the ground truth is "what Kazuhito's YOLOX model says". That model's training data is undisclosed and the [`audit_summary.csv`](file:///d:/hand-seal-dojo/data/audit/audit_summary.csv) shows the "Pure Naruto Hand Sign Data" external dataset has **85–96 % near-duplicate images** within its train split. High same-source accuracy ≠ real accuracy. |
| **Freeze and version the ONNX graph + its exact preprocessing** | Check in the ONNX opset version, input shape, normalization (currently: raw 0–255 BGR with 114-pad letterbox), and NMS params as a config file next to the model | Your web spike re-implements this from scratch in [JS postprocessing](file:///d:/hand-seal-dojo/web/spike/index.html#L98-L129). Any drift between the Python `YoloxONNX` and the JS port silently changes accuracy. The [parity test](file:///d:/hand-seal-dojo/web/spike/index.html#L163-L183) is a great start but it tests *images* — it doesn't cover the live camera path with mirror-flip and timing. |
| **Event-level metrics, not frame-level** | False-confirmations-per-minute, time-from-pose-to-confirm, per-seal recall, idle false-positive rate | Your [idle test](file:///d:/hand-seal-dojo/perception/service.py#L156-L163) counts false confirms but only in one self-run. There's no logged replay you can re-evaluate. |

### They would skip (or defer)

| Skip | Why |
|---|---|
| **Training your own model now** | The YOLOX nano model gets 97–100 % per-class on the (flawed) external dataset. The risk is not model accuracy on similar data; it's out-of-distribution failure with new people and cameras. Fix the evaluation gap first. |
| **Server-side inference** | For a public browser game there's no need for a backend GPU. YOLOX nano is 3.6 MB; ONNX Runtime Web can run it on-device. Your [web spike already proves this](file:///d:/hand-seal-dojo/web/spike/index.html#L144-L160). |
| **Multi-platform native apps** | A progressive web app covers desktop, Android and iOS Safari without a store submission. Unity WebGL or Flutter Web add download size and complexity with no gain for a webcam game. |
| **Real-time multiplayer** | Nice-to-have, but irrelevant until single-player detection is validated on strangers. |

---

## 2. Raw Material to Capture and Keep

Your current artefacts are *downstream products*. A practitioner keeps the raw evidence so that downstream artefacts can be regenerated when the pipeline changes.

### What to keep

| Raw material | Format | Why the processed artefact isn't enough |
|---|---|---|
| **Per-frame score vectors** | JSONL per session: `{ t_ms, yolox_scores: {seal: score, ...}, mp_landmarks: [[x,y,z]×21×2], mp_handedness: [...], expect, frame_index }` | Your [`StabilityGate`](file:///d:/hand-seal-dojo/perception/service.py#L60-L82) thresholds are currently hand-tuned constants. With raw vectors you can sweep `WINDOW`, `NEED`, `ARM_BELOW`, per-seal thresholds offline. |
| **Consented multi-person session video** | WebM or MP4, separate consent form per person, stored outside git (GCS / S3) | Needed for leave-one-person-out cross-validation. The [parity samples](file:///d:/hand-seal-dojo/data/parity_samples) are stills from the Kaggle dataset — they don't test the temporal gating path at all. |
| **The ONNX graph, its opset, and exact preprocessing spec** | Version-controlled config YAML next to the `.onnx` file | You depend on Kazuhito's `YoloxONNX` class hidden inside [`third_party/`](file:///d:/hand-seal-dojo/third_party/NARUTO-HandSignDetection). The Python and JS implementations must stay in lockstep; a spec file makes that auditable. |
| **Primary-source seal definitions** | Scanned manga panels or Databook pages, with a citation per seal | Currently the "correct" pose is whatever the YOLOX model recognises. For a *teaching* game, you need a ground truth that isn't circular. |
| **Event-level ground-truth labels** | Human-annotated CSV: `session_id, t_start_ms, t_end_ms, seal, rater_id, correct (bool)` | The [`audit_summary.csv`](file:///d:/hand-seal-dojo/data/audit/audit_summary.csv) tells you per-image YOLOX accuracy on a dataset with ~90 % near-duplicates. It does not tell you whether the game *confirms the right seal at the right time* in live play. |

### What to stop relying on as primary evidence

| Current artefact | Problem |
|---|---|
| Cropped reference snapshots in [`shared/reference/`](file:///d:/hand-seal-dojo/perception/service.py#L44) | These are auto-saved from one person. They're fine as UI thumbnails but they're not evaluation data. |
| The "Pure Naruto Hand Sign Data" external dataset | [85–96 % near-duplicate rate in train](file:///d:/hand-seal-dojo/data/audit/audit_summary.csv); all 540×960 (one phone camera); no train→test leakage detected but the test split is tiny (4–10 images). |
| Kazuhito's README as documentation of model behaviour | It doesn't disclose training data composition, augmentation, or known failure modes. |

---

## 3. Concrete Methods, Tools, and Licences

### A. On-device inference: browser and phones

| Runtime | Where | Notes | ⚠ What I haven't measured |
|---|---|---|---|
| **ONNX Runtime Web** (ort-web) | Browser, `wasm` or `webgpu` backend | You already use this in [`index.html`](file:///d:/hand-seal-dojo/web/spike/index.html#L15). `wasm` works everywhere; `webgpu` is faster where available. Your serve.py already sends the COOP/COEP headers needed for multi-threaded WASM. | I have not measured inference speed on any mobile device. Your spike has a benchmark mode — run it on ≥3 phones and record the numbers. |
| **MediaPipe Web (Hands/Hand Landmarker)** | Browser | The `@mediapipe/tasks-vision` npm package runs the `.task` model in WASM+WebGL. Replaces your Python MediaPipe path. | I have not measured MediaPipe Web latency on phones. Their published benchmarks claim ~10 ms on a Pixel 7, but your game runs YOLOX *and* MediaPipe together — total budget is unclear. |
| **TensorFlow Lite** / **Core ML** (if native) | Android / iOS | Only needed if you go native. For a browser-first game, skip this. | N/A — I recommend skipping native for now. |

**Concrete first step**: run your existing [`web/spike/index.html`](file:///d:/hand-seal-dojo/web/spike/index.html) benchmark on 3 real phones (one budget Android, one mid-range, one iPhone) via `serve.py`'s LAN URL. Record the Hz per 10 s window. If p50 total is > 80 ms (~12 fps) on the budget phone, you need to drop to 320×320 input or skip every other frame.

### B. Record-and-replay harness

**Goal**: tune `StabilityGate` parameters offline against labelled sessions.

```
Capture path (live):
  camera frame → YOLOX → score vector → MediaPipe → landmarks
  → append { t_ms, scores, landmarks, expect, frame_idx } to session.jsonl

Replay path (offline):
  load session.jsonl → for each param combo (window, need, arm_below, ...):
      run StabilityGate → produce confirm events → compare to ground-truth labels
      → compute false_confirms/min, time_to_confirm, per-seal recall
```

**Tool**: plain Python, no framework needed. The core is a `replay(session_path, gate_params) → event_list` function and a `score(events, labels) → metrics` function. Sweep with `itertools.product`.

### C. Event-level evaluation metrics

| Metric | Definition | Why it matters |
|---|---|---|
| **False confirmations / minute** | Number of `new_confirm=True` events during a labelled "idle" or "wrong seal" segment, divided by segment duration | Your [idle test](file:///d:/hand-seal-dojo/perception/service.py#L207-L306) measures this for one person; you need it per-person, per-device. |
| **Time to confirm (ms)** | Time from when the player first reaches the correct pose (human-labelled) to `new_confirm=True` | Measures responsiveness. Target: < 800 ms for a fun game, < 400 ms for competitive. |
| **Per-seal recall** | Fraction of labelled correct-pose windows where the system confirms within a timeout | Your [spike data shows Hare at 2 % correct@0.7](file:///d:/hand-seal-dojo/data/spike_yolox.csv#L2-L3) and Ram at 4.7 % — these seals are essentially broken for *your* hands. You need per-seal recall per person. |
| **Per-seal confusion matrix** | Which seal does the model think you're doing when you're doing Hare? | The YOLOX spike collects `top_confusion` but only for one person; need this across people. |

### D. Leave-one-person-out validation with confidence intervals

**Protocol**:
1. Collect N sessions from K people (K ≥ 5, ideally 10+).
2. For each person k: train/tune gate on all *other* people, evaluate on person k.
3. Per-person metrics: per-seal recall, false confirm rate, time-to-confirm.
4. Report: mean ± 95 % bootstrap CI across people.

**Implementation**: Since you're tuning the *gate* (not the YOLOX model), "training" means selecting `WINDOW`, `NEED`, thresholds. The ONNX model is frozen.

```python
from scipy.stats import bootstrap
import numpy as np

def ci_95(values):
    res = bootstrap((np.array(values),), np.mean, confidence_level=0.95,
                    n_resamples=10000, method='percentile')
    return res.confidence_interval.low, res.confidence_interval.high
```

Group by person; don't pool frames across people (a person with 10× more frames would dominate the average).

### E. Score calibration

The YOLOX output scores are not probabilities — they're `objectness × class_score`. A well-calibrated score means "when the model says 0.7, the pose is correct 70 % of the time."

**Method**: Platt scaling (logistic regression on raw score → is_correct). Fit on held-out labelled data, one model per seal if sample sizes allow. Plot reliability diagrams (expected vs. observed frequency per bin). Use `sklearn.calibration.calibration_curve` and `CalibratedClassifierCV`.

**Why**: your [per-seal thresholds](file:///d:/hand-seal-dojo/perception/service.py#L36-L38) (`Snake: 0.6`, `Hare: 0.5`) are hand-tuned. After calibration, a single threshold (e.g., 0.5 calibrated) works across seals.

### F. Licences that matter for a public release

| Component | Licence | Risk |
|---|---|---|
| **Kazuhito's NARUTO-HandSignDetection** (code + YOLOX nano ONNX model) | [MIT](file:///d:/hand-seal-dojo/third_party/NARUTO-HandSignDetection/LICENSE) | Low for code. **⚠ The model weights may have been trained on copyrighted images. The MIT licence covers the code, not necessarily the training data. I cannot verify the training data provenance.** |
| **ONNX Runtime Web** | MIT | Clean. |
| **MediaPipe** (hand landmarker .task) | Apache 2.0 | Clean. Attribute in NOTICE file. |
| **"Pure Naruto Hand Sign Data"** (Kaggle) | ⚠ Check the Kaggle dataset licence | Often CC-BY or similar, but some Kaggle datasets are "research only". **I have not checked this dataset's specific licence.** |
| **"Naruto" IP** (character names, seal names, jutsu names, visual designs) | ⚠ **Shueisha / Masashi Kishimoto copyright** | Using "Naruto" in your game's branding, or reproducing seal imagery from the manga, is a trademark/copyright risk. Fan games exist in a grey area. **I cannot provide legal advice; consult a lawyer if you plan a commercial or high-visibility release.** |
| **YOLOX architecture** | Apache 2.0 (Megvii) | Clean for the architecture. The *weights* are Kazuhito's, not Megvii's. |

---

## 4. Outside Evidence That Would Show It Works — Defined Before You Test

### Pass/fail acceptance criteria

| Test | Pass threshold | Rationale |
|---|---|---|
| **Held-out strangers**: ≥ 5 people never seen during tuning, on ≥ 3 distinct devices (1 laptop webcam, 1 Android phone, 1 iPhone) | Per-seal recall ≥ 80 % for all 12 seals, averaged across people | Below 80 % the game feels broken for that seal. Your current live data shows Hare and Ram at ~2–5 % on yourself — these must improve dramatically. |
| **False confirm rate during idle** | ≤ 1 false confirmation per 2 minutes of idle, per person | Your current idle test is self-only and one room. |
| **Time to confirm** | p50 ≤ 800 ms, p95 ≤ 1500 ms from correct pose onset | A 1.5 s delay makes the game sluggish. |
| **Two independent human raters** label each session's confirm events as correct/incorrect, comparing the player's pose to a primary-source reference image for each seal | Inter-rater agreement (Cohen's κ) ≥ 0.8 | If raters disagree, your ground truth is ambiguous and no model metric is meaningful. |
| **Browser inference speed on mobile** | p50 total frame time ≤ 80 ms (≥12 fps) on a mid-range 2023 Android phone | ⚠ **I have no data to predict this.** Your spike has the benchmark; run it. |

### What I cannot verify from this codebase

| Claim or assumption | Why I can't verify it |
|---|---|
| YOLOX nano generalises to non-Asian hands, dark skin tones, bright sunlight | The training data is undisclosed; the audit dataset appears to be one ethnicity, one camera. |
| The 12 zodiac seals in the model match the canonical Naruto sources | I have not seen primary-source comparisons. The model could be recognising Kazuhito's personal hand shapes. |
| Browser inference is fast enough on phones | I don't have phone benchmarks. Your spike logs them but none are checked in. |
| The "Pure Naruto Hand Sign Data" licence permits redistribution | I haven't read the Kaggle page. |
| Naruto IP usage is legally safe | This is a legal question, not a technical one. |

---

## 5. Platform Decision, Kill Spike, and First Thing to Build

### Decision: **Web (plain HTML + JS + ONNX Runtime Web)**

| Factor | Web | Unity WebGL | Flutter Web |
|---|---|---|---|
| **Camera access** | `getUserMedia` — universal, well-documented | Unity WebGL camera is awkward (plugin or JS bridge) | `camera` package — works but has rough edges on web |
| **ONNX inference** | `ort-web` directly; you've already proven it in [`index.html`](file:///d:/hand-seal-dojo/web/spike/index.html) | Barracuda / Sentis — ONNX support exists but is less battle-tested in WebGL | `onnxruntime` via FFI — very experimental on web |
| **Download size** | ~4 MB total (3.6 MB model + ort WASM) | 15–30 MB minimum for Unity WebGL loader + framework | 5–10 MB for Flutter engine + app |
| **Mobile browser** | Works today on Safari and Chrome | Unity WebGL is unreliable on mobile Safari; Chrome on Android is OK | Flutter Web on mobile is usable but not great |
| **Iteration speed** | Edit HTML/JS, refresh | C# compile → Unity build → test | Dart hot reload (good) but web platform has caveats |
| **You already have** | A working parity test and camera benchmark | A Unity project with a WebSocket client — couples you to a Python backend | Nothing |

> ⚠ I have not benchmarked Unity WebGL or Flutter Web inference on phones. My claim that "Unity WebGL is unreliable on mobile Safari" is based on widely-reported community experience, not my own measurement.

**Why not Unity**: Your current Unity project ([`game/HandSealDojo`](file:///d:/hand-seal-dojo/game/HandSealDojo)) is a thin WebSocket client that receives results from a Python server. Moving to browser-native inference *eliminates the server entirely*. That's the right architecture for a public, zero-install game. Unity WebGL would add 10-25 MB of download for rendering features you don't need in a webcam-overlay game.

**Why not Flutter**: Flutter Web's camera and ML inference support is less mature than the raw Web APIs you're already using. No evidence it would be better.

### One-day kill spike: "Prove web wrong"

**Goal**: find a concrete reason the web platform cannot ship.

**Morning (4 hours)**:
1. Take your existing [`web/spike/index.html`](file:///d:/hand-seal-dojo/web/spike/index.html) and add MediaPipe Hand Landmarker (via `@mediapipe/tasks-vision` CDN).
2. Run **both** YOLOX + MediaPipe on the same camera frames, measuring total frame time.
3. Test on 3 real phones:
   - One budget Android (e.g., Samsung A14, Redmi Note 12)
   - One mid-range Android (e.g., Pixel 7a)
   - One iPhone (Safari)
4. Record: total frame time p50/p95, whether `getUserMedia` works at all, whether the page crashes.

**Afternoon (4 hours)**:
5. Add a minimal `StabilityGate` in JS. Wire up a single-seal "confirm Tiger" flow with camera → YOLOX → gate → confirm event → visual feedback.
6. Ask one person who has *never seen the game* to try confirming Tiger, Dragon, and Snake. Note whether they succeed within 10 seconds per seal.

**Kill criteria** (if any is true, web is wrong):
- Total frame time > 150 ms (< 7 fps) on the mid-range Android.
- `getUserMedia` fails on any of the 3 devices.
- Tab crashes or runs out of memory within 2 minutes on any device.
- The test person cannot confirm any seal within 30 seconds.

If the spike survives, web is confirmed. If it fails, the failure tells you exactly what to fix (e.g., drop to YOLOX-only without MediaPipe, or reduce input resolution).

### First thing to build

**The record-and-replay harness.** Not the game UI, not the web port — the harness.

Why: every other decision (gate params, thresholds, per-seal difficulty tiers, whether you need MediaPipe at all) depends on being able to replay sessions and compute metrics offline. Right now your only feedback loop is running `service.py --idle-test` and watching yourself. That loop produced the current thresholds — which don't work for Hare or Ram even on your own hands.

Concretely:

1. **Add a `--record PATH` flag to [`service.py`](file:///d:/hand-seal-dojo/perception/service.py)** that appends one JSONL line per frame:
   ```json
   {"t_ms": 1234, "yolox": {"Rat": 0.12, "Tiger": 0.89, ...}, "mp_hands": 2, "mp_landmarks": [...], "expect": "Tiger", "frame_idx": 42}
   ```
2. **Write `replay.py`** that reads a session JSONL + a label CSV, runs `StabilityGate` with configurable params, and outputs event-level metrics (false confirms/min, time-to-confirm, per-seal recall).
3. **Record 3 sessions** of yourself doing all 12 seals, plus 2 minutes idle. Label them by hand (CSV with start/end times and correct seal).
4. **Sweep gate params** and print a table showing which combinations meet the pass thresholds from §4.

This gives you the evaluation infrastructure *before* you collect stranger data or port to the web — so when you do collect data, you have a machine to measure it.

---

## Summary of What Is and Isn't Verified

| Statement in this document | Status |
|---|---|
| YOLOX nano is MIT-licensed (code) | ✅ Verified — [LICENSE file](file:///d:/hand-seal-dojo/third_party/NARUTO-HandSignDetection/LICENSE) |
| YOLOX nano gets 97–100 % on the external dataset | ✅ Verified — [audit_summary.csv](file:///d:/hand-seal-dojo/data/audit/audit_summary.csv) |
| External dataset has 85–96 % near-duplicates | ✅ Verified — same CSV |
| External dataset has 0 train→test leakage | ✅ Verified — same CSV |
| Hare and Ram are broken on live webcam | ✅ Verified — [spike_yolox.csv](file:///d:/hand-seal-dojo/data/spike_yolox.csv) shows 2 % and 4.7 % correct@0.7 |
| MediaPipe often sees only 1 hand on two-hand seals | ✅ Verified — [spike_mediapipe.csv](file:///d:/hand-seal-dojo/data/spike_mediapipe.csv) and baseline show 0 % two-hands on several seals |
| ONNX Runtime Web runs YOLOX in-browser | ✅ Verified — [web spike code](file:///d:/hand-seal-dojo/web/spike/index.html) exists and includes parity test |
| Browser inference is fast enough on phones | ❌ **Not measured** — spike has a benchmark but no phone results are checked in |
| Unity WebGL is unreliable on mobile Safari | ❌ **Not measured** — community anecdote, not personal test |
| YOLOX training data is safe to redistribute | ❌ **Cannot verify** — training data is undisclosed |
| Naruto IP is safe to use in a public game | ❌ **Cannot verify** — legal question |
| MediaPipe Web + YOLOX can run together at ≥12 fps on a phone | ❌ **Not measured** — this is what the kill spike tests |
| Platt scaling will improve threshold consistency | ❌ **Not measured** — requires labelled calibration data you don't have yet |
