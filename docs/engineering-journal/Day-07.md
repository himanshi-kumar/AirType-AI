# Day 07

## Sprint

Sprint 6 — Voice Output (Text-to-Speech via pyttsx3)

---

## Objective

Give AirType AI a voice: enable touchless speech output by adding a `SPEAK` key that speaks typed sentences aloud using offline Text-to-Speech (TTS) on a non-blocking background thread.

---

## What I Built

### New Files

| File | Purpose |
|------|---------|
| `src/speech.py` | `Speaker` class wrapping `pyttsx3` with daemon threading and mutual exclusion lock |
| `tests/test_speech.py` | 13 unit tests covering initialization, non-blocking execution, safe stopping, and UI layout integration |

### Modified Files

| File | Changes |
|------|---------|
| `src/keyboard.py` | Added `SPEAK` key on bottom row between `SPC` and `BACK`, styled with dedicated purple color scheme (`COLOR_SPEAK_NORMAL`, `COLOR_SPEAK_HOVER`, `COLOR_SPEAK_BORDER`) |
| `src/main.py` | Wired `Speaker` instance into pipeline, intercepted `SPEAK` key pinch with highest priority, added cleanup on shutdown |
| `requirements.txt` | Added `pyttsx3==2.99` dependency |

---

## Concepts Learned

### 1. Offline Text-to-Speech Architecture

`pyttsx3` interfaces directly with operating system native speech synthesis engines:
- **macOS**: `NSSpeechSynthesizer` (via PyObjC)
- **Windows**: SAPI5
- **Linux**: eSpeak

Benefits:
- **Zero latency**: Instant synthesis without HTTP request roundtrips.
- **Offline & Private**: Audio generation occurs locally without cloud endpoints or API credentials.

### 2. Daemon Background Threading & Frame-Rate Preservation

`pyttsx3.runAndWait()` blocks until sentence playback concludes (typically 1–3 seconds). Calling this directly inside the 30fps OpenCV loop would freeze camera acquisition, causing 30–90 dropped frames.

Solution:
```python
self._thread = threading.Thread(
    target=self._speak_blocking,
    args=(text,),
    daemon=True
)
self._thread.start()
```

- `daemon=True` guarantees the worker thread terminates automatically when the parent process exits.
- Calling `stop()` cancels existing utterances before queuing fresh speech.

### 3. Thread Safety & Mutex Locking

`pyttsx3.Engine` instances are not thread-safe. A `threading.Lock` protects engine state during invocations, preventing race conditions or segmentation faults when users trigger rapid repeated pinches.

---

## Test Results

```
.venv/bin/python -m unittest tests.test_speech -v
Ran 13 tests in 7.605s — OK

.venv/bin/python -m unittest tests.test_spelling -v
Ran 30 tests in 0.018s — OK
```

---

## Full Pipeline (Sprint 6)

```
Frame
  ↓ flip
  ↓ MediaPipe inference → landmarks
  ↓ get_landmark_position(INDEX_TIP, THUMB_TIP)
  ↓ predictor.get_suggestions(typed_text)
  ↓ keyboard.set_suggestions(words)
  ↓ keyboard.draw(frame, finger_pos)
  ↓ pinch.update() → clicked?
      ├── hovered_suggestion? → select_suggestion()
      ├── hovered_key == "SPEAK"? → speaker.speak(typed_text)  ← NEW (Sprint 6)
      └── hovered_key?
            ├── SPC? → auto-correct last word + append space
            ├── BACK? → delete last char
            └── other → append letter
  ↓ imshow
```
