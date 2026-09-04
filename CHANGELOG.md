# Changelog

## v0.12.0

### Sprint 12 – Keypad Layout Modes & Visual Ripple Feedback

### Added

- **Keypad Layout Modes (`src/keyboard.py`)**: Stateful `ABC` (letters) and `123` (numbers, symbols, punctuation) layouts with dynamic toggle key and zero vertical layout shift.
- **Visual Ripple Effect (`src/keyboard.py`)**: `RippleEffect` class implementing an expanding and alpha-fading ring animation with bounded sub-frame ROI compositing on every keypress and suggestion selection.
- **CLR Key**: Instant text buffer reset key with dedicated coral-red aesthetic and audio cue.
- **Enhanced Sound Synthesis (`src/audio.py`)**: New in-memory synthesizers for `play_clear()` (descending D5+A4+D4 chord) and `play_mode_switch()` (C5+E5 chime).
- **Adaptive Key Typography**: Dynamic text size calculation ensuring multi-character labels (`SPEAK`, `BACK`, `CLR`, `123`, `ABC`) render cleanly within key borders without clipping.
- `tests/test_ripple.py` — 19 unit tests covering RippleEffect mathematics, layout toggling, CLR key behavior, and ripple pruning lifecycle.
- `docs/engineering-journal/Day-13.md` — Animation lifecycle, ROI alpha blending optimizations, and acoustic cue design.

### Learning Outcomes

- Parametric animation synthesis with sub-frame ROI blending in OpenCV
- Stateful virtual keyboard layout switching with invariant control baselines
- Multimodal feedback integration (tactile illusion via combined visual ripple + acoustic cues)

---

## v0.11.0

### Sprint 11 – Two-Hand Typing Support & Performance Counter

### Added

- **Multi-Hand Tracking (`src/hand_detector.py`)**: Upgraded default `max_hands` to 2; added `get_all_hands()` method returning landmark positions for each detected hand with dedicated per-hand `LandmarkSmoother` instances.
- **Dual Skeleton Visualization**: Render distinct colors for each detected hand (Cyan for hand 0, Magenta for hand 1) for clear visual disambiguation.
- **Independent Pinch State (`src/gesture.py`)**: New `MultiPinchDetector` class maintaining independent `PinchDetector` rising-edge state and cooldowns per hand channel.
- **Multi-Finger Visuals & Cursors (`src/keyboard.py`)**: `Keyboard.draw()` accepts `finger_positions: list` supporting simultaneous hover highlights across hands. Added live FPS counter display to the typing stats bar.
- **Multi-Hand Main Event Loop (`src/main.py`)**: Simultaneous key processing, independent pinch dispatch, and rolling 30-frame window FPS estimator.
- `tests/test_multihand.py` — 13 unit tests covering multi-hand pinch detection, independent cooldowns, simultaneous pinch, multi-cursor rendering, and FPS counter.
- `docs/engineering-journal/Day-12.md` — Multi-hand architectural design, concurrency considerations, and rolling-window FPS profiling.

### Learning Outcomes

- Multi-agent state isolation: independent channel management for low-latency concurrent inputs
- Multi-channel input filtering with per-channel EMA signal processors
- Rolling-window frequency estimation for robust real-time FPS telemetry

---

## v0.10.0

### Sprint 10 – Landmark Smoothing (EMA Jitter Elimination)

### Added

- **`src/smoothing.py`**: New `LandmarkSmoother` class implementing Exponential Moving Average (EMA) filter with configurable α parameter (default 0.45).
- **Transparent Integration**: EMA smoothing applied inside `hand_detector.py`'s `get_landmark_position()` — zero changes required in main.py or keyboard.py.
- **Per-Landmark Independence**: Each of MediaPipe's 21 landmark IDs is tracked and smoothed independently.
- **Automatic Reset**: Smoother history clears when hand disappears to prevent stale coordinate artifacts.
- `tests/test_smoothing.py` — 17 unit tests covering EMA convergence, jitter reduction, step response, formula correctness, independent tracking, reset behavior, and alpha validation.
- `docs/engineering-journal/Day-11.md` — EMA theory, α tuning, and EMA vs Kalman Filter comparison.

### Learning Outcomes

- Digital signal processing: Exponential Moving Average as a low-pass filter
- Step response analysis and convergence rate: (1-α)^n residual decay
- Transparent Decorator pattern for non-invasive coordinate filtering

---



### Sprint 9 – Optimized Autocorrect (BK-Tree, Bayesian Ranking & Expanded Vocabulary)

### Added

- **BK-Tree Index**: Burkhard-Keller tree data structure for O(log N) fuzzy search, replacing O(N) brute-force vocabulary scanning. Prunes 85-95% of candidates using the triangle inequality.
- **Bayesian Frequency-Weighted Ranking**: Scoring formula `score = edit_distance - α × log(frequency + 1)` breaks edit-distance ties in favor of common words (e.g., "THE" wins over "THY").
- **Expanded Vocabulary**: Grew from ~300 to ~2,500 words covering ~85% of everyday English text, including technology, nature, food, professions, finance, and education domains.
- **Length Pre-Filtering**: Skips candidate words with impossible length differences before computing expensive edit distances.
- `tests/test_bktree.py` — 32 unit tests covering BK-Tree construction, search correctness (brute-force comparison), Bayesian ranking, length filtering, vocabulary validation, and backward compatibility.
- `docs/engineering-journal/Day-10.md` — BK-Tree theory, Bayesian autocorrect, and metric tree concepts.

### Learning Outcomes

- BK-Tree (Burkhard-Keller Tree) metric indexing and triangle inequality pruning
- Bayesian / Noisy Channel Model for spelling correction
- `__slots__` optimization for memory-efficient tree nodes
- Length bucketing as a cheap pre-filter before expensive DP computation

---

## v0.8.0

### Sprint 8 – Premium UI Polish (Glassmorphism, Live Stats & Speaking Pulse)

### Added

- **Glassmorphism Keys**: Keys rendered with semi-transparent alpha blending (`cv2.addWeighted`) for a modern frosted-glass HUD aesthetic.
- **Live Typing Stats Bar**: Displays current word count and live WPM (Words Per Minute) above the suggestion bar.
- **Visual SPEAK Pulse**: Smooth 2Hz sine-animated orange pulse ring around the `SPEAK` key during active TTS speech.
- `tests/test_ui.py` — 9 unit tests covering glassmorphism ROI blending, stats calculations, and pulse rendering.
- `docs/engineering-journal/Day-09.md` — UI compositing and parametric animation principles.

### Learning Outcomes

- OpenCV ROI alpha compositing with `cv2.addWeighted`
- Time-based parametric visual animations in 30fps game loops
- Real-time typing speed and telemetry calculation

---

## v0.7.0

### Sprint 7 – Sound Feedback (Synthesized Audio via NumPy + sounddevice)

### Added

- `src/audio.py` — `SoundPlayer` class with 5 synthesized audio cues (no audio files)
- 5 distinct sounds mapped to 5 distinct keyboard events:
  - Letter key → 440Hz sine click (50ms)
  - `SPC` → 220Hz sine thud (80ms)
  - `BACK` → 330→165Hz descending sweep (70ms)
  - Suggestion selected → 523→784Hz rising sweep (120ms)
  - `SPEAK` activated → 440+550Hz chord (150ms)
- Fade-out envelope on all sounds to eliminate end-click artifacts
- `tests/test_audio.py` — 18 unit tests (synthesis length, amplitude, dtype, fade, playback)
- `docs/engineering-journal/Day-08.md` — audio fundamentals and design decisions

### Learning Outcomes

- Digital audio sample representation (44,100 Hz, float32 in `[-1.0, 1.0]`)
- Sine wave synthesis from NumPy (`sin(2πft)`)
- Frequency sweep (chirp) synthesis via phase integration
- Chord synthesis via superposition + normalization
- Fade-out envelope to prevent speaker pop artifacts
- Pre-computation pattern for real-time audio in a 30fps loop

---

## v0.6.0

### Sprint 6 – Voice Output (Offline Text-to-Speech via pyttsx3)

### Added

- `src/speech.py` — `Speaker` class with non-blocking background daemon thread and mutex locking
- `SPEAK` key on bottom row of virtual keyboard (centered between `SPC` and `BACK`)
- Distinct purple styling for `SPEAK` key (`COLOR_SPEAK_NORMAL`, `COLOR_SPEAK_HOVER`, `COLOR_SPEAK_BORDER`)
- Priority event dispatching in `main.py` routing pinches on `SPEAK` to `Speaker.speak()`
- `tests/test_speech.py` — 13 unit tests verifying initialization, non-blocking execution, stopping, and layout
- `docs/engineering-journal/Day-07.md` — Sprint 6 architectural design and concurrency notes
- `pyttsx3==2.99` dependency added to `requirements.txt`

### Learning Outcomes

- Operating system native speech engines (`NSSpeechSynthesizer`, SAPI5, eSpeak)
- Daemon threads for background task execution without blocking OpenCV 30fps stream
- Mutual exclusion locks (`threading.Lock`) for thread-unsafe C-extension APIs

---

## v0.5.0

### Sprint 5 – Spelling Auto-Correct (Damerau-Levenshtein + QWERTY Weights)

### Added

- `KEY_CENTERS` — QWERTY key center pixel coordinates for distance weighting
- `get_substitution_cost()` — Euclidean distance-based substitution cost between keys
- `damerau_levenshtein_distance()` — QWERTY-weighted edit distance with transposition support
- `WordPredictor.get_autocorrect()` — finds closest vocabulary match for misspelled words
- Fuzzy backoff in `_predict_completions()` — fills suggestion slots with DL matches when prefix matches are insufficient
- Auto-correct on SPACE press — `register_click(predictor)` corrects the last word before appending space
- `tests/test_spelling.py` — 30 unit tests covering all new functionality
- `docs/engineering-journal/Day-06.md` — Sprint 5 design decisions and concepts

### Architecture

- `prediction.py` owns all spelling correction logic (Single Responsibility)
- `keyboard.py` calls predictor on SPC press (dependency injection, not import)
- `main.py` passes predictor into keyboard click handler (wiring, no logic)
- First-letter mismatch penalty (+1.5) prevents unrelated short words from polluting corrections

### Learning Outcomes

- Levenshtein distance (edit distance DP algorithm)
- Damerau-Levenshtein extension (transposition = 80% of human typos)
- QWERTY-weighted substitution cost (physical keyboard distance)
- Fuzzy string matching backoff (prefix → edit distance)
- Threshold scaling (longer words → more tolerance for typos)

---

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