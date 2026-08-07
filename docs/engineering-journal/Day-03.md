# Day 03

## Sprint

Sprint 2 — MediaPipe Hand Detection

---

## Objective

Integrate MediaPipe hand detection into the AirType AI pipeline.
Debug the `AttributeError: module 'mediapipe' has no attribute 'solutions'` blocker.

---

## The Problem (Root Cause Analysis)

### Error

```
AttributeError: module 'mediapipe' has no attribute 'solutions'
```

### Investigation

```bash
python -c "import mediapipe as mp; print(mp.__version__); print(hasattr(mp,'solutions'))"
# Output:
# 0.10.35
# False
```

### Root Cause

MediaPipe released two completely different APIs across its version history:

| API | Versions | Import | Status |
|-----|----------|--------|--------|
| **Legacy Solutions API** | `0.8.x` – `0.9.x` | `mp.solutions.hands` | ❌ Removed in 0.10.x |
| **Modern Tasks API** | `0.10.x+` | `mediapipe.tasks.python.vision` | ✅ Active |

Every tutorial on YouTube and StackOverflow uses the OLD API.
Our installed version `0.10.35` is the NEW API.
This is a breaking change — the module was simply removed.

### Why Did Google Change It?

The old `solutions` API was a monolithic design:
- Everything bundled together in one giant module
- Hard to test, maintain, or extend
- Not portable to mobile/edge devices

The new Tasks API is modular:
- Each capability (hands, face, pose) is an independent `.task` bundle
- The bundle contains model weights + metadata + pre/post-processing
- Portable: same API works on Python, Android, iOS, Web

---

## The Fix

### What Changed

| Before (broken) | After (fixed) |
|----------------|---------------|
| `mp.solutions.hands` | `mediapipe.tasks.python.vision.HandLandmarker` |
| `mp_hands.Hands(...)` | `vision.HandLandmarker.create_from_options(options)` |
| `mp.solutions.drawing_utils` | Custom `_draw_landmarks()` and `_draw_connections()` |
| No model file needed | `assets/hand_landmarker.task` (7.5MB bundle) |

### New Architecture

```
Webcam frame (BGR NumPy array)
        ↓
  cv2.cvtColor → RGB
        ↓
  mp.Image wrapper (MediaPipe container)
        ↓
  detector.detect_for_video(mp_image, timestamp_ms)
        ↓
  HandLandmarkerResult
    └── hand_landmarks: List[List[NormalizedLandmark]]
                                    ↓
                    landmark.x * width  →  pixel_x
                    landmark.y * height →  pixel_y
                                    ↓
                          cv2.circle + cv2.line
```

---

## Concepts Learned

### 1. Why BGR → RGB Conversion Is Required

OpenCV was built in the early 2000s to read Windows Bitmap files.
Windows stored color channels in BGR order for hardware alignment reasons.
OpenCV kept this convention.

MediaPipe's models were trained on RGB images.
If you pass BGR directly, the model sees red as blue and blue as red.
This causes wrong predictions or no detections.

One line fixes everything:
```python
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

### 2. Why MediaPipe Uses Normalized Coordinates [0.0–1.0]

Instead of returning `x=640, y=240`, MediaPipe returns `x=0.5, y=0.5`.

**Why?** The model was trained on hundreds of thousands of images of different
resolutions. By normalizing coordinates, the model becomes **resolution-independent**.
The same trained model works on a 480p webcam and a 4K camera without retraining.

**We scale back to pixels only when drawing:**
```python
cx = int(landmark.x * frame_width)
cy = int(landmark.y * frame_height)
```

### 3. The Two-Stage MediaPipe Pipeline

MediaPipe Hands uses two models internally:

**Stage 1: Palm Detector**
- Fast, lightweight CNN
- Runs on the full (downscaled) frame
- Detects a bounding box around the palm
- Does NOT detect individual fingers yet

**Stage 2: Hand Landmark Model**
- Detailed, crops to the bounding box from Stage 1
- Detects exactly 21 landmarks
- Runs only on the palm region → faster, more accurate

**Why two stages?**
Running a detailed landmark model on the entire frame every frame is expensive.
The palm detector is fast and cheap. Once we know WHERE the hand is,
we zoom in and run the expensive model only on that small crop.
This is the **Cascade architecture** pattern used in most real-time CV systems.

### 4. Why 21 Landmarks?

The human hand has:
- 1 wrist
- 5 fingers × 4 joints = 20 points
- Total = 21 points

MediaPipe maps each anatomical point:

```
        8   12   16   20
        |    |    |    |
    4   7   11   15   19
    |   |    |    |    |
    3   6   10   14   18
    |   |    |    |    |
    2   5    9   13   17
     \  |    |    |   /
          0 (WRIST)
```

### 5. RunningMode.VIDEO vs LIVE_STREAM

| Mode | Behavior | Use When |
|------|----------|----------|
| `IMAGE` | One image, one result, independent | Batch processing photos |
| `VIDEO` | Synchronous, timestamps required | Real-time webcam (our case) |
| `LIVE_STREAM` | Asynchronous, callback-based | High-performance production |

We use `VIDEO` because:
- Synchronous = easier to understand
- Result is available immediately after the function call
- No callbacks, no threading complexity

We will upgrade to `LIVE_STREAM` in Sprint 5.

### 6. The Kalman Filter (Why Timestamps Matter)

MediaPipe uses a Kalman filter to smooth landmark positions between frames.

A Kalman filter is a mathematical prediction algorithm that:
1. Uses the current detected position (noisy, from the model)
2. Uses physics (velocity = how fast the hand moved between frames)
3. Combines them to produce a smoothed, stable position

**The timestamp tells the filter how much time passed.**
If you pass timestamp=0 every frame, the filter thinks no time passed,
and its predictions become unstable. Always increment the timestamp.

---

## Bugs Faced

### Bug 1: `AttributeError: module 'mediapipe' has no attribute 'solutions'`

Root cause: MediaPipe 0.10.x removed the legacy Solutions API.
Fix: Migrated to the Tasks API (see above).

### Bug 2: `cd main.py` — wrong command

`cd` only works on directories, not files.
To run a Python file: `python3 main.py` or `python3 src/main.py`.

### Bug 3: Running `import mediapipe as mp` directly in the terminal

The terminal is a shell (zsh/bash), not a Python interpreter.
Python code can only run inside Python. To test a quick import:
```bash
python3 -c "import mediapipe as mp; print(mp.__version__)"
```

---

## Files Changed

| File | Change |
|------|--------|
| `src/hand_detector.py` | Complete rewrite: Tasks API + drawing helpers |
| `assets/hand_landmarker.task` | New: ML model bundle (downloaded 7.5MB) |
| `CHANGELOG.md` | Updated with v0.3.0 |

---

## Reflection

Today I debugged a real breaking change in an open-source library — exactly
the kind of problem engineers face in production. The lesson:
**never trust a tutorial's import statement blindly.** Always check which API
version your installed package actually exposes.

The fix required understanding the architecture of the new API deeply enough
to rewrite the integration from scratch. That deeper understanding is worth
more than a tutorial that just "worked."

The 21-landmark model is now running on my webcam in real time.
Next sprint: finger tracking and key collision detection.
