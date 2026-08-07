# Day 04

## Sprint

Sprint 3 — Finger Tracking, Hover Detection, Pinch-to-Click Typing

---

## Objective

Make the keyboard interactive:
- Highlight the key the index finger is hovering over
- Detect a pinch gesture (index + thumb) as a "click"
- Type the character when a pinch fires over a key
- Display the typed text on screen

---

## What I Built

### New Files

| File | Purpose |
|------|---------|
| `src/gesture.py` | `PinchDetector` class — pure gesture math |

### Modified Files

| File | Changes |
|------|---------|
| `src/hand_detector.py` | Implemented `get_landmark_position()`, added `_last_results` cache |
| `src/keyboard.py` | `Key.contains()` collision detection, hover highlight, text buffer, SPACE + BACK |
| `src/main.py` | Full orchestrator pipeline: detect → get positions → draw → pinch → type |

---

## Concepts Learned

### 1. Euclidean Distance (Pinch Measurement)

To measure if two fingertips are "close enough" to count as a pinch,
we compute the straight-line distance between them.

```
d = √( (x2-x1)² + (y2-y1)² )
```

This is the Pythagorean theorem applied to 2D pixel coordinates.
Both fingertips are points on a grid (the video frame).
The distance between them is the hypotenuse of the right triangle
formed by their horizontal and vertical separations.

**Why not Manhattan distance?**
Manhattan = |Δx| + |Δy|. This only works for grid-aligned paths.
Pinch is a diagonal motion — Euclidean captures it correctly.

### 2. Rising-Edge Detection

A pinch held for 1 second at 30fps would fire 30 characters without this.

**Rising edge** = fire ONLY on the frame the pinch STARTS, not while held.

```
Frame N-1: not pinching   →  _was_pinching = False
Frame N:   pinching       →  was False + is True NOW = FIRE ✅
Frame N+1: still pinching →  was True + is True = don't fire ❌
```

This pattern is borrowed from electronics. Every button in every app
uses some version of this — it's one of the most important UX patterns
in interactive software.

### 3. Cooldown (Debounce)

After a click fires, we ignore all further presses for 20 frames (~0.66s).

This prevents:
- Stutter-clicks when the finger trembles near the threshold
- Accidentally typing 3 letters when you meant 1

In embedded systems, this is called **debouncing** (a hardware capacitor
smooths a noisy button signal). In software, we simulate it with a counter.

### 4. AABB Collision Detection

AABB = Axis-Aligned Bounding Box.

To check if a finger (a point) is over a key (a rectangle):

```python
x <= point.x <= x + width
AND
y <= point.y <= y + height
```

This works because our keys are always straight (not rotated).
If they were rotated, we'd need the Separating Axis Theorem (SAT).

### 5. Caching Results (Performance Pattern)

`detect()` runs ML inference every frame (expensive ~15ms).
`get_landmark_position()` needed the same data but should NOT re-run ML.

Solution: store the result on `self._last_results`. Any method can read it
for free. This pattern is called **memoization** (caching a computation result).

### 6. The Orchestrator Pattern

`main.py` contains ZERO logic. It only:
- Creates components
- Calls them in the right order
- Passes data between them

Each component is independently testable:
- `gesture.py` needs only two tuples — no camera
- `keyboard.py` needs only a frame and a point — no MediaPipe
- `hand_detector.py` needs only a frame — no keyboard

---

## Pipeline Diagram

```
Webcam frame
    ↓
cv2.flip() — mirror
    ↓
HandDetector.detect() — ML inference, draws skeleton
    ↓
get_landmark_position(INDEX_FINGER_TIP)  ← index_tip = (x, y) or None
get_landmark_position(THUMB_TIP)         ← thumb_tip = (x, y) or None
    ↓
Keyboard.draw(finger_pos=index_tip)
    ├── get_hovered_key(index_tip) — AABB collision check against 28 keys
    ├── key.draw(hovered=True/False) — green highlight if hovered
    └── _draw_text_display() — show typed string above keyboard
    ↓
PinchDetector.update(index_tip, thumb_tip)
    ├── euclidean_distance() — is distance < 40px?
    ├── rising-edge check — is this the first frame of the pinch?
    └── cooldown check — have 20 frames passed since the last click?
    ↓
if clicked: Keyboard.register_click()
    └── append letter / space / backspace to typed_text
    ↓
cv2.imshow() — display
```

---

## Key Learning

Three algorithms drive the entire typing experience:

1. **Euclidean distance** → detect pinch
2. **AABB** → detect which key the finger is over
3. **Rising edge + cooldown** → fire exactly one click per pinch

All three are first-year math, applied to real Computer Vision.

---

## Reflection

Today the keyboard became real. Hovering the index finger over Q highlights it
in green. Pinching fires a character. Holding BACK deletes.

The most important insight: keep logic separated. `gesture.py` doesn't know
about keyboards. `keyboard.py` doesn't know about MediaPipe. `main.py` doesn't
know about math. Each file can be read, tested, and changed independently.

That is clean architecture.
