# Day 13 — Sprint 12: Keypad Layout Modes & Visual Ripple Feedback

## What I Built

In Sprint 12, I added two major capabilities to AirType AI:
1. **Dynamic Keypad Modes (`ABC` ↔ `123`) & `CLR` Key**: A full numbers and symbols layout with punctuation, arithmetic symbols, and brackets, plus a dedicated `CLR` button to reset typed text.
2. **Visual Ripple Animation (`RippleEffect`)**: An expanding, fading circular shockwave originating from the center of any clicked key or suggestion box, giving instantaneous visual confirmation of pinch registrations.
3. **Dedicated Audio Signatures**: Custom harmonic chords for the `CLR` key (descending D5→A4→D4 reset chord) and `123`/`ABC` mode toggle (C5+E5 major third chime).

---

## Animation Lifecycle: The `RippleEffect` Class

Physical keycaps provide tactile feedback through key travel and bottoming out. In a touchless air keyboard, visual feedback must substitute for tactile sensation.

### Mathematical Formulation

The ripple is parameterized purely by real time $t$:

$$\text{progress}(t) = \min\left(1.0, \frac{t - t_0}{T}\right)$$

Where $t_0$ is key activation timestamp and $T = 0.30\text{ s}$ is total duration.

1. **Expanding Radius**:
   $$r(t) = R_{\max} \cdot \text{progress}(t)$$
   As time increases from $0$ to $T$, radius grows linearly from $0$ to $R_{\max} = 45\text{ px}$.

2. **Linear Opacity Fade**:
   $$\alpha(t) = \alpha_{\max} \cdot (1.0 - \text{progress}(t))$$
   The ring begins opaque ($\alpha_{\max} = 0.80$) and fades to complete transparency ($\alpha = 0.0$) at $t = t_0 + T$.

### Performance Optimization: Bounding Box Slicing

Drawing full-frame OpenCV overlays with `cv2.addWeighted` on $1280 \times 720$ frames would consume unnecessary memory bandwidth (~2.7MB per frame per ripple). 

Instead, `RippleEffect.draw()` computes an exact sub-frame bounding box (Region of Interest) around the circle:

$$x_1 = \max(0, c_x - r - \text{pad}), \quad x_2 = \min(W, c_x + r + \text{pad})$$
$$y_1 = \max(0, c_y - r - \text{pad}), \quad y_2 = \min(H, c_y + r + \text{pad})$$

Only this miniature $\sim 100 \times 100$ patch is copied, blended, and written back to the frame. When the duration elapses, the ripple is automatically pruned from `keyboard.ripples`.

---

## Keypad Layout Architecture

To accommodate numbers, special characters, and punctuation without crowding the screen or adding additional rows, we introduced **stateful layout modes**:

```
Mode: "ABC" (Letter Layout)
Row 0:  Q  W  E  R  T  Y  U  I  O  P
Row 1:  A  S  D  F  G  H  J  K  L
Row 2:  Z  X  C  V  B  N  M
Row 3: [123] [  SPC  ] [SPEAK] [ BACK ] [CLR]

Mode: "123" (Numbers & Symbols Layout)
Row 0:  1  2  3  4  5  6  7  8  9  0
Row 1:  @  #  $  %  &  *  (  )  !  ?
Row 2:  .  ,  :  ;  '  "  -  +  =  /
Row 3: [ABC] [  SPC  ] [SPEAK] [ BACK ] [CLR]
```

### Consistent Interaction Invariant

Row 3 remains structurally invariant across both modes:
- Both layouts maintain `SPC`, `SPEAK`, `BACK`, and `CLR` on the same row with identical dimensions.
- The mode switch key resides at the bottom-left, toggling between `123` and `ABC`.
- All keys maintain their exact vertical baseline ($y = 626\text{ px}$), ensuring zero layout shift for core navigation controls.

---

## Audio Design

| Action | Frequency / Profile | Duration | Psychological Cue |
| :--- | :--- | :--- | :--- |
| **Normal Key** | 440 Hz (A4) Sine | 50 ms | Crisp tactile key click |
| **Space** | 220 Hz (A3) Sine | 80 ms | Low acoustic thud (heavy spacebar) |
| **Backspace** | 330 → 165 Hz Sweep | 70 ms | Descending chirp ("undo") |
| **Suggestion** | 523 → 784 Hz Sweep | 120 ms | Ascending reward chime |
| **Speak** | 440 + 550 Hz Chord | 150 ms | Rich modal confirmation |
| **Mode Switch** | 523 + 659 Hz (C5+E5) | 100 ms | Light harmonious notification |
| **Clear** | 587 + 440 + 293 Hz (D5+A4+D4) | 180 ms | Full descending triad (destructive reset) |

---

## Verification & Metrics

- **Unit Test Coverage**: 19 new tests in `tests/test_ripple.py`, bringing the total project test suite to **152 tests passing (100%)**.
- **Frame Rate**: The optimized ROI alpha blending maintains steady 28–30 FPS even when both hands trigger ripples simultaneously.
