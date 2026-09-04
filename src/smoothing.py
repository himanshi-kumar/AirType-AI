"""
smoothing.py — Sprint 10: Exponential Moving Average (EMA) Landmark Smoothing

WHY THIS FILE EXISTS
--------------------
Webcam-based hand tracking has inherent jitter: even when holding your hand
perfectly still, the detected landmark coordinates fluctuate by 2-5 pixels
per frame due to camera noise, lighting changes, and model uncertainty.

This jitter makes targeting small keys (58×50 pixels) frustrating — the
finger cursor "vibrates" and may jump between adjacent keys unexpectedly.

SOLUTION: Exponential Moving Average (EMA)
--------------------------------------------
EMA is a simple digital signal processing filter that smooths a noisy
time series by blending each new observation with a weighted history:

    smoothed[t] = α × raw[t] + (1 - α) × smoothed[t-1]

Where:
    α = smoothing factor (0.0 to 1.0)
    raw[t] = noisy observation at time t (raw landmark coordinate)
    smoothed[t-1] = previous smoothed value

INTUITION
---------
- α = 1.0 → No smoothing (output = raw input). Maximum responsiveness.
- α = 0.5 → 50% new data, 50% history. Good balance.
- α = 0.2 → 20% new data, 80% history. Very smooth but laggy.
- α = 0.0 → Output never changes (frozen). Useless.

Think of it as a "trust dial":
- High α = "I trust the new sensor reading a lot"
- Low α  = "I mostly trust where I thought the hand was before"

WHY NOT A KALMAN FILTER?
--------------------------
A Kalman filter is more sophisticated: it models the system dynamics
(velocity, acceleration) and measurement noise. For our use case:
- EMA is simpler (2 multiplications vs matrix algebra per frame)
- EMA has zero tuning beyond one parameter (α)
- EMA performs nearly as well for slow hand movements
- Kalman would only help for fast hand motions (which are rare during typing)

For a production system with gesture recognition (swipes, circles), upgrading
to Kalman or a 1D Butterworth filter would be worthwhile. For keyboard typing,
EMA is the right tool.

DEFAULT α = 0.45
------------------
Empirically tuned:
- α = 0.3: Too laggy — cursor trails behind the finger noticeably.
- α = 0.45: Sweet spot — jitter eliminated, cursor feels responsive.
- α = 0.6: Still jittery on lower-quality webcams.

The optimal value depends on webcam FPS and noise characteristics.
Users can adjust via the alpha parameter.

ARCHITECTURE DECISION
----------------------
LandmarkSmoother is a standalone class (not embedded in HandDetector) so it
can be unit-tested independently with synthetic data. HandDetector calls
smoother.update() inside get_landmark_position() — the rest of the system
sees no change in API. This is the Decorator pattern.
"""


class LandmarkSmoother:
    """
    Applies Exponential Moving Average (EMA) smoothing to hand landmark
    coordinates to eliminate per-frame jitter from webcam noise.

    USAGE
    ------
    smoother = LandmarkSmoother(alpha=0.45)

    # Each frame:
    raw_x, raw_y = detector.get_raw_landmark(frame, INDEX_FINGER_TIP)
    smooth_x, smooth_y = smoother.update(INDEX_FINGER_TIP, raw_x, raw_y)

    ATTRIBUTES
    ----------
    alpha : float
        Smoothing factor (0.0 to 1.0). Higher = more responsive, less smooth.
    _history : dict[int, tuple[float, float]]
        Maps landmark_id → last smoothed (x, y). Initialized on first call.
    """

    def __init__(self, alpha: float = 0.45):
        """
        Initialize the smoother.

        Parameters
        ----------
        alpha : float
            Smoothing factor in [0.0, 1.0].
            - 0.45 = recommended for 30fps webcam (good balance)
            - Higher = faster response, more jitter
            - Lower  = smoother cursor, more input lag
        """
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0.0, 1.0], got {alpha}")
        self.alpha = alpha
        self._history: dict = {}

    def update(self, landmark_id: int, raw_x: int, raw_y: int) -> tuple:
        """
        Apply EMA smoothing to a single landmark's coordinates.

        FIRST FRAME BEHAVIOR
        ----------------------
        On the first call for a given landmark_id, there's no history.
        We initialize the smoothed position to the raw position (no lag).
        Subsequent calls blend the new reading with the history.

        FORMULA
        --------
        smoothed_x = α × raw_x + (1 - α) × prev_smoothed_x
        smoothed_y = α × raw_y + (1 - α) × prev_smoothed_y

        Parameters
        ----------
        landmark_id : int
            The MediaPipe landmark ID (0-20). Each landmark is tracked
            independently so finger tip smoothing doesn't affect wrist.
        raw_x : int
            Raw x coordinate from MediaPipe (pixels).
        raw_y : int
            Raw y coordinate from MediaPipe (pixels).

        Returns
        -------
        tuple[int, int]
            Smoothed (x, y) coordinates rounded to integers.
        """
        if landmark_id not in self._history:
            # First observation: initialize with raw value (no lag)
            self._history[landmark_id] = (float(raw_x), float(raw_y))
            return (raw_x, raw_y)

        prev_x, prev_y = self._history[landmark_id]

        # EMA formula: blend new reading with history
        smooth_x = self.alpha * raw_x + (1 - self.alpha) * prev_x
        smooth_y = self.alpha * raw_y + (1 - self.alpha) * prev_y

        # Store the float values for next iteration (preserve precision)
        self._history[landmark_id] = (smooth_x, smooth_y)

        # Return integer pixel coordinates for drawing
        return (int(round(smooth_x)), int(round(smooth_y)))

    def reset(self, landmark_id: int = None) -> None:
        """
        Clear smoothing history.

        Parameters
        ----------
        landmark_id : int | None
            If provided, reset only that landmark.
            If None, reset ALL landmarks (e.g., when hand disappears).
        """
        if landmark_id is not None:
            self._history.pop(landmark_id, None)
        else:
            self._history.clear()

    @property
    def tracked_landmarks(self) -> list:
        """Return list of landmark IDs currently being tracked."""
        return list(self._history.keys())
