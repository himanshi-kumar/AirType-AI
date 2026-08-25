# Day 08

## Sprint

Sprint 7 — Sound Feedback (Synthesized Audio via sounddevice + NumPy)

---

## Objective

Give every AirType AI keyboard interaction an auditory signature using synthesized sounds generated entirely in-memory with NumPy — no audio files required.

---

## What I Built

### New Files

| File | Purpose |
|------|---------|
| `src/audio.py` | `SoundPlayer` class synthesizing 5 distinct sounds from sine/sweep/chord functions |
| `tests/test_audio.py` | 18 unit tests covering synthesis correctness, amplitude safety, fade-out, and playback |

### Modified Files

| File | Changes |
|------|---------|
| `src/main.py` | Initialized `SoundPlayer`; dispatched matching sound after every pinch event |

---

## Sound Design Map

| Event | Frequency | Duration | Technique |
|-------|-----------|----------|-----------|
| Letter keypress | 440Hz (A4) | 50ms | Pure sine |
| Spacebar | 220Hz (A3) | 80ms | Pure sine (lower = heavier) |
| Backspace | 330→165Hz | 70ms | Frequency sweep (falling) |
| Suggestion select | 523→784Hz (C5→G5) | 120ms | Frequency sweep (rising) |
| SPEAK activation | 440+550Hz | 150ms | Two-frequency chord |

---

## Concepts Learned

### 1. Digital Audio Fundamentals

Digital audio represents sound as discrete amplitude samples over time. At 44,100 Hz sample rate, each second of audio = 44,100 float values in `[-1.0, 1.0]`.

A pure tone at frequency `f`:
```
y[t] = A × sin(2π × f × t)
```
Where `t = [0, 1/44100, 2/44100, ...]` is the time axis.

### 2. Frequency Sweeps (Chirps)

A sweep slides from `f_start` to `f_end` linearly by integrating the instantaneous frequency:
```
phase(t) = 2π × (f_start × t + (f_end - f_start) × t² / (2 × duration))
y(t)    = A × sin(phase(t))
```
Rising sweeps (523→784Hz) feel like "reward" or "completion."
Falling sweeps (330→165Hz) feel like "undo" or "cancel."

This psychological mapping (rising = positive, falling = negative) is used in:
- Game sound design (level-up = rising arpeggio, death = descending tone)
- Phone UIs (SMS sent = rising chime, error = descending buzz)
- AirType AI: suggestion = rising, backspace = falling

### 3. Chord Synthesis via Superposition

The principle of superposition: overlaying multiple waves simply sums them.
```
chord(t) = A × (sin(2π × f1 × t) + sin(2π × f2 × t)) / 2
```
Dividing by N (number of frequencies) prevents clipping to values > 1.0.

### 4. Fade-Out Envelope (Preventing Click Artifacts)

When audio playback stops abruptly at a non-zero sample, the speaker membrane makes a discontinuous jump → audible "click" pop.

A linear fade applied to the final 10ms eliminates this:
```python
fade_curve = np.linspace(1.0, 0.0, fade_samples)
samples[-fade_samples:] *= fade_curve
```

This is called an "envelope" in audio synthesis — modulating amplitude over time.

### 5. Pre-Computation vs On-Demand Synthesis

All sounds are synthesized ONCE in `__init__`, stored as arrays.
In the real-time 30fps loop, playback = `sd.play(array)` — one system call.

The alternative (synthesize on each keypress) would add ~1ms of NumPy computation per pinch — acceptable, but unnecessary. Pre-computation is a standard technique in audio engines.

---

## Test Results

```
.venv/bin/python -m unittest tests.test_audio tests.test_speech tests.test_spelling -v
Ran 60 tests in 1.367s — OK

Sprint 7 audio:   18 tests ✅
Sprint 6 speech:  13 tests ✅  (regression)
Sprint 5 spelling: 30 tests ✅ (regression)
```

---

## Full Pipeline (Sprint 7)

```
Frame
  ↓ flip
  ↓ MediaPipe inference → landmarks
  ↓ predictor.get_suggestions(typed_text) [fuzzy DL backoff if needed]
  ↓ keyboard.set_suggestions(words)
  ↓ keyboard.draw(frame, finger_pos)
  ↓ pinch.update() → clicked?
      ├── hovered_suggestion?
      │     ├── keyboard.select_suggestion()
      │     └── audio.play_suggestion()      ← rising chime (Sprint 7)
      ├── hovered_key == "SPEAK"?
      │     ├── speaker.speak(typed_text)
      │     └── audio.play_speak()           ← chord (Sprint 7)
      └── hovered_key?
            ├── SPC?  → auto-correct + append space + audio.play_space()   ← (Sprint 7)
            ├── BACK? → delete char          + audio.play_backspace()       ← (Sprint 7)
            └── letter → append             + audio.play_keypress()        ← (Sprint 7)
  ↓ imshow
```
