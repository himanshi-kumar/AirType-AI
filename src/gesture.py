"""
gesture.py — Sprint 3: Pinch Gesture Detection

WHY THIS FILE EXISTS
--------------------
Gestures are pure mathematics. They don't care about webcam, keyboard, or UI.
Keeping gesture logic in its own file follows the Single Responsibility Principle:
  gesture.py   → "Is a pinch happening?" (math only)
  keyboard.py  → "What does a pinch DO on the keyboard?"
  main.py      → "Orchestrate everything"

This separation means you can unit-test gestures with just numbers,
no camera or UI required.

THE PINCH GESTURE
-----------------
A pinch = index finger tip and thumb tip are close together.

Mathematically: distance(index_tip, thumb_tip) < threshold_pixels

We use Euclidean distance:
    d = √((x2-x1)² + (y2-y1)²)

This is the straight-line distance between two 2D points —
the same formula from high school geometry (Pythagorean theorem).
"""

import math


# ── Tuning constants ──────────────────────────────────────────────────────────
# PINCH_THRESHOLD: how many pixels apart the fingertips can be and still
# count as a pinch. 40px is roughly the width of a fingernail on a 640px frame.
# Too low  (e.g. 20px) → hard to trigger, frustrating UX
# Too high (e.g. 80px) → accidental triggers constantly
PINCH_THRESHOLD = 40

# CLICK_COOLDOWN_FRAMES: after a click registers, how many frames to ignore
# further clicks. At 30fps, 20 frames = ~0.66 seconds between letters.
# This prevents one pinch from registering as 10 key presses.
CLICK_COOLDOWN_FRAMES = 20


def euclidean_distance(point_a: tuple, point_b: tuple) -> float:
    """
    Calculate the straight-line distance between two 2D pixel points.

    ALGORITHM: Pythagorean theorem
    --------------------------------
    Given points A=(x1, y1) and B=(x2, y2):

        d = √( (x2-x1)² + (y2-y1)² )

    This works because pixels live on a 2D grid (like an x-y coordinate plane).
    The distance between two pixels IS the hypotenuse of a right triangle
    whose legs are Δx (horizontal distance) and Δy (vertical distance).

    WHY NOT MANHATTAN DISTANCE?
    ----------------------------
    Manhattan distance = |x2-x1| + |y2-y1|
    It measures distance as if you can only walk in straight lines (like a city grid).
    Pinch is a DIAGONAL motion — the fingers come together at any angle.
    Euclidean captures diagonal distances correctly. Manhattan does not.

    Parameters
    ----------
    point_a : (int, int)
        Pixel coordinates of the first point (e.g. index finger tip).
    point_b : (int, int)
        Pixel coordinates of the second point (e.g. thumb tip).

    Returns
    -------
    float
        Distance in pixels. Always ≥ 0.
    """
    dx = point_b[0] - point_a[0]   # horizontal separation
    dy = point_b[1] - point_a[1]   # vertical separation
    return math.sqrt(dx * dx + dy * dy)


class PinchDetector:
    """
    Detects when a pinch gesture starts (used as a "click" trigger).

    WHY A CLASS (not just a function)?
    ------------------------------------
    A function can detect "is a pinch happening right now?"
    But we need MORE than that:
        - detect EXACTLY WHEN the pinch STARTS (the press moment)
        - prevent holding a pinch from firing 30 clicks per second
        - track cooldown state between frames

    State across frames = class. Stateless math = function.
    PinchDetector wraps the stateless euclidean_distance() with stateful
    cooldown tracking.

    ATTRIBUTES
    ----------
    threshold : int
        Max pixel distance to count as a pinch. Default: PINCH_THRESHOLD.
    cooldown : int
        Frames remaining before the next click can register. 0 = ready.
    _was_pinching : bool
        Was a pinch active in the PREVIOUS frame?
        Used to detect the RISING EDGE (start of pinch, not duration).
    """

    def __init__(self, threshold: int = PINCH_THRESHOLD):
        self.threshold = threshold
        self.cooldown = 0            # frames until next click is allowed
        self._was_pinching = False   # pinch state in the previous frame

    def update(self, index_tip, thumb_tip) -> bool:
        """
        Call this every frame. Returns True ONLY on the frame the pinch starts.

        RISING EDGE DETECTION
        ----------------------
        We want to fire on the MOMENT the pinch starts — not while held down.
        This is called "rising edge" detection (borrowed from electronics):

            Frame N-1: no pinch   (_was_pinching = False)
            Frame N:   pinch!     (_was_pinching = False) → FIRE ✅
            Frame N+1: still pinch (_was_pinching = True)  → don't fire ❌

        Without this, holding a pinch for 1 second at 30fps would register 30
        characters. With rising-edge detection, it registers exactly 1.

        COOLDOWN
        ---------
        After a click fires, we set cooldown = CLICK_COOLDOWN_FRAMES.
        Each frame decrements it by 1. Until it reaches 0, no new clicks.
        This handles the case where the finger naturally "bounces" near the
        threshold — we don't want stutter-clicks.

        Parameters
        ----------
        index_tip : (int, int) | None
            Pixel position of index finger tip. None if no hand detected.
        thumb_tip : (int, int) | None
            Pixel position of thumb tip. None if no hand detected.

        Returns
        -------
        bool
            True = a new click event fired this frame. False otherwise.
        """
        # Decrement cooldown every frame regardless of pinch state
        if self.cooldown > 0:
            self.cooldown -= 1

        # If no hand detected, reset pinch state and return False
        if index_tip is None or thumb_tip is None:
            self._was_pinching = False
            return False

        # Measure current distance between fingertips
        distance = euclidean_distance(index_tip, thumb_tip)
        is_pinching_now = distance < self.threshold

        # Rising-edge detection:
        # Fire ONLY if:  pinching NOW  AND  was NOT pinching last frame  AND  not in cooldown
        clicked = (
            is_pinching_now
            and not self._was_pinching
            and self.cooldown == 0
        )

        if clicked:
            self.cooldown = CLICK_COOLDOWN_FRAMES

        # Update state for the next frame's edge detection
        self._was_pinching = is_pinching_now

        return clicked

    @property
    def is_pinching(self) -> bool:
        """
        Read-only property: is a pinch currently held down?

        WHY A PROPERTY?
        ---------------
        Using @property lets callers write:
            if detector.is_pinching:  ...
        instead of:
            if detector.is_pinching():  ...

        It looks like an attribute (clean), but runs a method (flexible).
        Python properties are the standard way to expose computed read-only state.
        """
        return self._was_pinching
