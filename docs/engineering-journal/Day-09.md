# Day 09

## Sprint

Sprint 8 — Premium UI Polish (Glassmorphism, Audio Pulse, and Live Stats)

---

## Objective

Elevate the AirType AI visual experience to professional standards: replace opaque key boxes with glassmorphic semi-transparent key overlays, add real-time typing analytics (word count and dynamic WPM), and introduce a pulsing audio visualizer ring around the SPEAK key during TTS playback.

---

## What I Built

### New Files

| File | Purpose |
|------|---------|
| `tests/test_ui.py` | 9 unit tests verifying glassmorphic rendering, stats bar calculation/drawing, and pulse ring animation |

### Modified Files

| File | Changes |
|------|---------|
| `src/keyboard.py` | Implemented `cv2.addWeighted` ROI blending on `Key.draw()`, added `_draw_stats_bar()` with WPM metrics, and `_draw_speak_pulse()` with 2Hz sine wave oscillation |
| `src/main.py` | Tracked session start time and passed `is_speaking` + `session_start` to `keyboard.draw()` |

---

## Concepts Learned

### 1. Glassmorphism via OpenCV Alpha Blending

Traditional computer vision interfaces often draw solid colored rectangles over camera frames, which obscures fingers and ambient surroundings. Glassmorphism blends a solid overlay with the camera feed:

$$\text{dst}(x,y) = \alpha \cdot \text{overlay}(x,y) + (1-\alpha) \cdot \text{roi}(x,y) + \gamma$$

Using `cv2.addWeighted(overlay, 0.55, roi, 0.45, 0)` delivers a frosted glass aesthetic while preserving finger visibility beneath keys.

### 2. Time-Driven Parametric Animations

Visual animations inside a real-time OpenCV loop cannot rely on `time.sleep()`. Instead, animations are parameterized by continuous wall-clock time:

$$r(t) = r_{\text{base}} + A \cdot \sin(2\pi f t)$$

Where $f = 2.0\text{ Hz}$ and $A = 6\text{ px}$. This produces a smooth, non-blocking breathing pulse around the active SPEAK button regardless of frame rate variations.

### 3. Live Typing Analytics (Words Per Minute)

WPM is calculated as standard in typing tutors:

$$\text{WPM} = \frac{\text{Word Count}}{\Delta t_{\text{minutes}}}$$

To avoid erratic spikes during the first few seconds of typing, WPM display activates after 10 seconds of elapsed typing time.

---

## Test Results

```
.venv/bin/python -m unittest tests.test_ui tests.test_audio tests.test_speech tests.test_spelling -v
Ran 69 tests in 1.376s — OK

Sprint 8 UI:       9 tests ✅
Sprint 7 Audio:   18 tests ✅
Sprint 6 Speech:  13 tests ✅
Sprint 5 Spelling: 30 tests ✅
```

---

## Complete Full-Stack Pipeline (Sprint 8)

```
Frame (1280x720)
  ↓ flip & mirror
  ↓ MediaPipe Hand Landmark Inference
  ↓ Extract index_tip, thumb_tip
  ↓ WordPredictor: NLP bigram & Damerau-Levenshtein backoff
  ↓ Keyboard.set_suggestions()
  ↓ Keyboard.draw():
      ├── Stats Bar (Words + WPM)
      ├── Word Prediction Boxes
      ├── Typed Text Bar
      ├── Glassmorphic Translucent Keys (addWeighted)
      └── SPEAK Pulse Ring (when Speaker.is_speaking)
  ↓ PinchDetector.update() → Click Dispatch
      ├── Suggestion Click → Auto-complete + Rising Chime
      ├── SPEAK Click → Trigger TTS Thread + Activation Chord
      └── Character / SPC / BACK → Auto-correct / Type / Delete + Sound FX
  ↓ cv2.imshow
```
