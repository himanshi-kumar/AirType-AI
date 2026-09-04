# Day 12 — Sprint 11: Two-Hand Typing Support

## What I Built

Added full two-hand tracking to AirType AI. Both hands can now hover over
and press keys independently, with color-coded skeletons, per-hand cooldowns,
and an FPS counter to monitor performance.

## The Architecture: Per-Hand Everything

The core challenge of multi-hand typing is **independence**: each hand must
operate as a completely separate typing channel. This required changes at
every layer of the pipeline:

### Layer 1: Hand Detection (`hand_detector.py`)
- Changed `max_hands` from 1 to 2.
- Added `get_all_hands(frame)` returning a list of hand dicts.
- Each hand gets its own `LandmarkSmoother` instance.
- Skeleton colors differentiate hands: cyan (hand 0) vs magenta (hand 1).

### Layer 2: Gesture Detection (`gesture.py`)
- Created `MultiPinchDetector` that manages independent `PinchDetector`
  instances per hand index.
- Each hand has its own cooldown timer and rising-edge state.
- `tick_inactive()` cleans up smoothers for hands that disappear.

### Layer 3: Keyboard Rendering (`keyboard.py`)
- `draw()` now accepts `finger_positions: list` — all fingers from all hands.
- Any key under ANY finger gets the hover highlight.
- Added `fps: float` parameter for the stats bar FPS counter.

### Layer 4: Main Loop (`main.py`)
- Rewrote the loop to iterate over `detector.get_all_hands()`.
- Each hand's pinch is processed independently via `multi_pinch.update()`.
- FPS measured via a rolling window of 30 frame timestamps.

## Key Design Decision: Why Not Share Cooldowns?

If two hands shared a single PinchDetector, pinching with the left hand
would block the right hand for 20 frames (~0.66 seconds). This would make
two-hand typing slower than one-hand typing — the opposite of the goal.

By giving each hand its own detector, they're truly independent:
- Left hand presses 'H', enters 20-frame cooldown.
- Right hand immediately presses 'I' — not blocked.
- Result: "HI" typed in 2 frames instead of 22 frames.

## FPS Counter: Rolling Window Method

Instead of measuring per-frame `1/dt` (noisy), we use a rolling window:

```python
timestamps = [t1, t2, ..., t30]  # last 30 frame times
fps = 29 / (t30 - t1)            # 29 intervals over their total span
```

This produces a stable reading that updates smoothly. At 30fps with
two-hand detection, typical performance is 22-28fps depending on hardware.

## Performance Impact

Two-hand detection roughly doubles MediaPipe inference time because:
- Palm detector runs once (finds all hands in frame).
- Landmark model runs TWICE (once per detected hand).

On a modern laptop (M1/M2 Mac, Intel i5+):
- One hand: ~30fps
- Two hands: ~22-28fps (still smooth for typing)

## Test Results

131 tests passing (13 new Sprint 11 tests + 118 existing).
Key tests:
- `test_hand0_cooldown_doesnt_block_hand1`: Independence guarantee.
- `test_draw_with_finger_positions_list`: Multi-cursor rendering.
