# Day 11 — Sprint 10: Exponential Moving Average (EMA) Landmark Smoothing

## What I Built

Added a real-time digital signal processing filter to eliminate per-frame
jitter on hand landmark coordinates, making air-typing significantly more
precise and less frustrating.

## The Problem: Webcam Jitter

Even when holding your hand perfectly still in front of the camera, MediaPipe's
detected landmark coordinates fluctuate by 2–5 pixels per frame. This is caused by:

1. **Camera sensor noise**: Random pixel-level variation in each frame.
2. **Lighting changes**: Micro-fluctuations in ambient light affect edge detection.
3. **Model uncertainty**: The ML model's confidence varies slightly frame-to-frame.

On a 58×50 pixel key, ±5px jitter means the "cursor" may unintentionally
hover over adjacent keys, causing mistyped letters.

## The Solution: Exponential Moving Average (EMA)

```
smoothed[t] = α × raw[t] + (1 - α) × smoothed[t-1]
```

Where α ∈ (0, 1] is the smoothing factor.

### Intuition: The "Trust Dial"

Think of α as how much you trust the latest sensor reading:
- **α = 1.0**: "I completely trust the new reading" → no smoothing, raw pass-through.
- **α = 0.5**: "I trust the new reading 50%, and my previous estimate 50%."
- **α = 0.2**: "I mostly trust where I thought the hand was, and only slightly adjust."

### Why α = 0.45?

I empirically tested several values:
- α = 0.3: Cursor trails behind the finger — typing feels sluggish.
- α = 0.45: Jitter eliminated, cursor feels responsive and direct.
- α = 0.6: Still some jitter visible on lower-quality webcams.

## Architecture: Transparent Integration

The `LandmarkSmoother` class is a standalone module (`src/smoothing.py`)
that gets instantiated inside `HandDetector.__init__()`. The smoothing
happens inside `get_landmark_position()`:

```
Raw MediaPipe coordinates → EMA filter → Smoothed coordinates
                                         ↑ returned to main.py
```

**Key design decisions**:
- Skeleton drawing uses RAW coordinates (no visual lag on the skeleton).
- Hover/collision detection uses SMOOTHED coordinates (stable targeting).
- Each landmark ID is tracked independently (index finger smoothing doesn't
  affect thumb coordinates).
- When the hand disappears, the smoother history is cleared to avoid
  stale positions when the hand reappears.

## EMA vs Kalman Filter

| Feature | EMA | Kalman Filter |
| :--- | :--- | :--- |
| Parameters to tune | 1 (α) | 4+ (Q, R, initial state, model) |
| Computation per frame | 2 multiplications | Matrix algebra (6×6) |
| Handles velocity | No | Yes (predicts next position) |
| Best for | Slow, steady movements | Fast, dynamic gestures |
| Implementation | 15 lines | 80+ lines |

For keyboard typing (slow, deliberate finger movements), EMA is optimal.
For future gesture recognition (swipes, circles), a Kalman filter would
track fast trajectories better.

## Test Results

118 tests passing (17 new Sprint 10 tests + 101 existing tests).
Key tests:
- `test_jitter_reduction`: Proves smooth output has lower variance than jittery input.
- `test_step_response_convergence`: Proves cursor catches up within ~10 frames.
- `test_independent_tracking`: Proves per-landmark independence.
