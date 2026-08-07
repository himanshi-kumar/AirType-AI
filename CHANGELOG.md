# Changelog

## v0.4.0

### Sprint 3 – Finger Tracking, Hover Detection, Pinch-to-Click Typing

### Added

- `src/gesture.py` — `PinchDetector` class with rising-edge detection and cooldown
- `Key.contains()` — AABB collision detection (is a finger over this key?)
- `Key.draw(hovered)` — green highlight when the index finger is over the key
- `Keyboard.get_hovered_key()` — linear search over all keys using `contains()`
- `Keyboard.register_click()` — appends letter / space / backspace to `typed_text`
- `Keyboard._draw_text_display()` — shows typed text with cursor above keyboard
- `HandDetector.get_landmark_position()` — returns pixel coords of any landmark
- `HandDetector._last_results` — caches ML result so landmarks can be read without re-inference
- SPACE and BACK special keys

### Architecture

- `main.py` is now a pure orchestrator (zero logic, only wiring)
- Each module independently testable (gesture.py needs only tuples, no camera)

### Learning Outcomes

- Euclidean distance formula (Pythagorean theorem applied to 2D pixels)
- Rising-edge detection (fire on pinch START, not while held)
- Cooldown / debounce (prevent stutter-clicks)
- AABB collision detection (point-in-rectangle test)
- Memoization / caching (reuse expensive computation across method calls)
- Orchestrator pattern (main.py coordinates without containing logic)

---

## v0.3.0

### Sprint 2 – Hand Detection (MediaPipe Tasks API)

### Added

- `hand_detector.py` — `HandDetector` class using MediaPipe Tasks API
- `assets/hand_landmarker.task` — compiled ML model bundle (7.5MB)
- 21-landmark skeleton drawn live on webcam feed
- Color-coded landmarks: green = index tip, yellow = thumb, blue = wrist
- Integrated `HandDetector` into `main.py` pipeline

### Fixed

- **`AttributeError: module 'mediapipe' has no attribute 'solutions'`**
  - Root cause: `mediapipe >= 0.10.x` removed the legacy `mp.solutions` API
  - Fix: migrated to the modern `mediapipe.tasks` Tasks API
  - `mp.solutions.hands.Hands()` → `vision.HandLandmarker.create_from_options()`
  - `mp.solutions.drawing_utils` → custom `_draw_landmarks()` and `_draw_connections()`

### Learning Outcomes

- Why MediaPipe uses normalized [0,1] coordinates instead of pixels
- How the two-stage pipeline works: Palm Detector → Landmark Model
- Why MediaPipe Tasks API requires a `.task` model bundle file
- `RunningMode.VIDEO` vs `RunningMode.LIVE_STREAM` (sync vs async)
- Why `detect_for_video()` needs a monotonically increasing timestamp
- How a Kalman filter smooths landmark positions between frames

---

## v0.2.0

### Sprint 1 – Webcam Engine

### Added

- Connected webcam using OpenCV
- Read live frames using VideoCapture
- Displayed webcam stream
- Learned OpenCV frame lifecycle
- Understood images as NumPy arrays
- Learned image shape `(Height, Width, Channels)`
- Learned BGR color representation

### Learning Outcomes

- Difference between pixels and landmarks
- Why OpenCV uses BGR
- How a webcam continuously streams frames
- Importance of `waitKey()`
- Resource cleanup using `release()`