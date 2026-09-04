"""
hand_detector.py — Sprint 11: Two-Hand Typing Support

WHAT'S NEW IN SPRINT 11
-----------------------
Sprint 2:  MediaPipe Tasks API hand landmarker wrapper
Sprint 10: + EMA landmark smoothing
Sprint 11: + Two-hand detection (max_hands=2)
           + Per-hand color coding (cyan for Left, magenta for Right)
           + get_all_hands() returns landmark data for ALL detected hands
           + Per-hand independent EMA smoothers

WHY THIS FILE EXISTS
--------------------
MediaPipe >= 0.10.x dropped the old `mp.solutions` API.
The modern replacement is the MediaPipe Tasks API, which is:
  - More modular (each task is independent)
  - More performant (uses optimized .task model bundles)
  - More production-ready (supports LIVE_STREAM mode with callbacks)

This file wraps that API into a clean, reusable HandDetector class.

ARCHITECTURE DECISION
---------------------
We use RunningMode.VIDEO instead of LIVE_STREAM because:
  - VIDEO mode is synchronous: call → get result immediately
  - LIVE_STREAM is asynchronous: result arrives via callback
  - For a beginner project, synchronous is simpler and equally fast
  - We can upgrade to LIVE_STREAM in Sprint 5 for performance
"""

import os
import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from smoothing import LandmarkSmoother


# ---------------------------------------------------------------------------
# LANDMARK IDS (MediaPipe's 21-point hand skeleton)
# ---------------------------------------------------------------------------
# MediaPipe always returns exactly 21 landmarks per hand.
# Each landmark has (x, y, z) where x and y are normalized [0.0 to 1.0].
# You multiply by frame width/height to get pixel coordinates.
#
# Key IDs you will use constantly in this project:
#   0  = WRIST
#   4  = THUMB_TIP
#   8  = INDEX_FINGER_TIP   ← your primary "pointer"
#   12 = MIDDLE_FINGER_TIP
#   16 = RING_FINGER_TIP
#   20 = PINKY_TIP
# ---------------------------------------------------------------------------
INDEX_FINGER_TIP = 8
THUMB_TIP = 4
WRIST = 0


class HandDetector:
    """
    Detects hands in a video frame and draws landmarks using MediaPipe Tasks API.

    WHY A CLASS?
    ------------
    The detector holds internal state: the compiled model, a frame counter,
    and drawing options. State + behavior = class. This follows the Single
    Responsibility Principle: this class ONLY does hand detection.

    ATTRIBUTES
    ----------
    detector : vision.HandLandmarker
        The compiled MediaPipe HandLandmarker instance.
    timestamp_ms : int
        A monotonically increasing counter. VIDEO mode requires each frame
        to have a unique timestamp so the tracker knows time has progressed.
    HAND_CONNECTIONS : set
        The 20 bone connections of the 21-point hand skeleton.
    _last_results : HandLandmarkerResult | None
        Cached result from the most recent detect() call.
        Allows get_landmark_position() to be called without re-running ML.
    """

    def __init__(
        self,
        model_path: str = "assets/hand_landmarker.task",
        max_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
    ):
        """
        Initialize the HandDetector — compile the model and configure it.

        WHY THESE PARAMETERS?
        ---------------------
        max_hands=2
            Sprint 11: We track two hands for split-keyboard typing.
            Both left and right hands can hover over and press keys
            independently, doubling typing speed potential.

        min_detection_confidence=0.7
            Palm detector threshold. Below 0.7 = too many false positives
            (detects non-hands). Above 0.9 = misses partially visible hands.
            0.7 is the industry sweet spot.

        min_tracking_confidence=0.7
            Once a hand is found, the tracker re-uses the previous frame's
            landmark prediction unless this threshold isn't met. Higher =
            more stable but slower to recover when the hand moves fast.

        WHY model_path IS A PARAMETER (not hardcoded)
        -----------------------------------------------
        If you hardcode a path, you cannot unit-test this class without the
        real file present. Making it a parameter = testable, configurable,
        and reusable across environments (local, CI, Docker).
        """

        # Resolve path relative to the project root (one level up from src/)
        abs_model_path = os.path.join(
            os.path.dirname(__file__), "..", model_path
        )
        abs_model_path = os.path.abspath(abs_model_path)

        if not os.path.exists(abs_model_path):
            raise FileNotFoundError(
                f"\n[HandDetector] Model not found at: {abs_model_path}\n"
                "Download it with:\n"
                "  curl -L -o assets/hand_landmarker.task \\\n"
                "    https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )

        # ── Step 1: BaseOptions ─────────────────────────────────────────────
        # Tells MediaPipe WHERE the compiled model lives.
        # The .task file bundles the model weights + metadata + pre/post
        # processing into a single portable file (like a zip for ML models).
        base_options = python.BaseOptions(model_asset_path=abs_model_path)

        # ── Step 2: HandLandmarkerOptions ───────────────────────────────────
        # Configures the detector's behavior at compile time (not runtime).
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            # VIDEO mode = synchronous, frame-by-frame. Each detect_for_video()
            # call blocks until the result arrives. Simple and reliable.
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # ── Step 3: Create the detector ─────────────────────────────────────
        # create_from_options() compiles and loads the model into memory.
        # This is expensive — done ONCE in __init__, never inside detect().
        self.detector = vision.HandLandmarker.create_from_options(options)

        # Frame counter: VIDEO mode needs a strictly increasing timestamp.
        # 33ms ≈ one frame at 30fps. This tells the Kalman-filter tracker
        # how much time passed between frames for motion prediction.
        self.timestamp_ms = 0

        # HAND_CONNECTIONS: a set of Connection(start, end) objects.
        # Each Connection = one "bone" in the hand skeleton (20 total).
        # Used in _draw_connections() to draw the skeleton lines.
        self.HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS

        # Cache the most recent inference result.
        # WHY CACHE IT?
        # detect() runs ML inference (expensive). get_landmark_position()
        # needs the same data but should NOT trigger a second inference.
        # By storing the result on self, any method can read it for free.
        self._last_results = None

        # ── Sprint 10: EMA Landmark Smoother ──────────────────────────────
        # Applies Exponential Moving Average to raw landmark coordinates
        # to eliminate 2-5px per-frame jitter from webcam noise.
        # α = 0.45: good balance between responsiveness and smoothness.
        # The smoother is applied inside get_landmark_position() only,
        # so drawn landmarks show raw positions (no visual lag on skeleton)
        # but the coordinates used for key hover/collision are smoothed.
        self._smoother = LandmarkSmoother(alpha=0.45)

        # ── Sprint 11: Per-hand smoothers for multi-hand support ────────
        # Each detected hand gets its own smoother instance keyed by
        # hand index (0 or 1). This prevents one hand's smoothing
        # history from interfering with the other's coordinates.
        self._hand_smoothers: dict = {}

        # ── Sprint 11: Per-hand skeleton colors ────────────────────────
        # Different colors help the user distinguish which hand is which.
        # Cyan = left hand (cool tones), Magenta = right hand (warm tones).
        self._hand_colors = [
            {  # Hand 0 colors
                "skeleton": (0, 200, 200),     # cyan
                "index": (0, 255, 0),          # green
                "thumb": (0, 255, 255),        # yellow
                "wrist": (255, 100, 0),        # blue
                "other": (255, 255, 255),      # white
            },
            {  # Hand 1 colors
                "skeleton": (200, 0, 200),     # magenta
                "index": (0, 200, 255),        # orange
                "thumb": (255, 0, 255),        # magenta
                "wrist": (255, 0, 100),        # pink
                "other": (220, 200, 255),      # light purple
            },
        ]

    def detect(self, frame) -> "np.ndarray":
        """
        Detect hands in one BGR frame, draw landmarks, return the frame.

        WHY RETURN FRAME?
        -----------------
        The caller (main.py) holds one frame reference. Returning the same
        frame (mutated in place) keeps the API clean: one in, one out.
        It also avoids copying a large NumPy array (640×480×3 = ~900KB).

        INTERNAL PIPELINE
        -----------------
        BGR frame
          ↓ cv2.cvtColor (channel swap)
        RGB frame
          ↓ mp.Image (MediaPipe container)
        MediaPipe image
          ↓ detector.detect_for_video (ML inference)
        21 NormalizedLandmarks per hand
          ↓ _landmark_to_pixel (scale to frame size)
        Pixel coordinates
          ↓ cv2.line / cv2.circle
        Frame with skeleton overlay drawn
        """

        # ── Step 1: BGR → RGB ───────────────────────────────────────────────
        # OpenCV captures in BGR (historical Windows bitmap quirk from 1990s).
        # MediaPipe's model was trained on RGB images — wrong channel order
        # = wrong predictions. This one line fixes the whole problem.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Step 2: Wrap in mp.Image ────────────────────────────────────────
        # MediaPipe Tasks API requires its own Image container, not raw NumPy.
        # mp.ImageFormat.SRGB = "this is standard RGB with 3 channels".
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # ── Step 3: Advance timestamp ───────────────────────────────────────
        # VIDEO mode tracks motion across frames. It needs a timestamp (in ms)
        # that always increases. 33ms = ~30fps. The tracker uses this to
        # predict where the hand "should" be in the next frame — this is what
        # makes landmark drawing smooth even when the model is uncertain.
        self.timestamp_ms += 33

        # ── Step 4: Run inference ───────────────────────────────────────────
        # detect_for_video() runs the full MediaPipe pipeline:
        #   1. Palm Detector (fast, low-res) → finds hand bounding box
        #   2. Landmark Model (detailed, crops to bounding box) → 21 points
        # Returns a HandLandmarkerResult with .hand_landmarks
        results = self.detector.detect_for_video(mp_image, self.timestamp_ms)

        # ── Step 5: Cache and draw ───────────────────────────────────────────
        # Store the result so get_landmark_position() can read it without
        # re-running the expensive ML inference.
        self._last_results = results

        # results.hand_landmarks is a list of lists:
        #   Outer list: one entry per detected hand (we configured max_hands=1)
        #   Inner list: 21 NormalizedLandmark objects (one per landmark)
        if results.hand_landmarks:
            h, w, _ = frame.shape  # get pixel dimensions once

            for hand_idx, hand_landmarks in enumerate(results.hand_landmarks):
                # Pick color scheme based on hand index (Sprint 11)
                colors = self._hand_colors[hand_idx % len(self._hand_colors)]
                # Draw skeleton lines FIRST so landmark dots appear on top
                self._draw_connections(frame, hand_landmarks, w, h, colors)
                # Draw landmark dots
                self._draw_landmarks(frame, hand_landmarks, w, h, colors)

        return frame

    # ────────────────────────────────────────────────────────────────────────
    # PRIVATE HELPER METHODS
    # Convention: prefix with _ means "internal — do not call from outside".
    # Callers should only use detect() and (later) get_landmark_position().
    # ────────────────────────────────────────────────────────────────────────

    def _landmark_to_pixel(self, landmark, width: int, height: int):
        """
        Convert normalized [0.0–1.0] coordinates to pixel coordinates.

        WHY NORMALIZED COORDINATES?
        ----------------------------
        MediaPipe returns x=0.5, y=0.3 instead of x=640, y=240.
        This makes the model resolution-independent — it produces identical
        results on a 480p webcam or a 4K camera. We scale to pixels at
        the last possible moment, only when drawing to screen.

        Example:
            landmark.x = 0.5, landmark.y = 0.3, width=640, height=480
            → pixel = (320, 144)
        """
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        return x, y

    def _draw_landmarks(self, frame, landmarks, width: int, height: int,
                        colors: dict = None):
        """
        Draw a colored circle at each of the 21 landmark positions.

        VISUAL DESIGN CHOICES (Sprint 11 — per-hand colors)
        ---------------------
        INDEX_FINGER_TIP (ID 8) → green (hand 0) or orange (hand 1)
        THUMB_TIP (ID 4) → yellow (hand 0) or magenta (hand 1)
        WRIST (ID 0) → blue (hand 0) or pink (hand 1)
        All others → white (hand 0) or light purple (hand 1)
        """
        if colors is None:
            colors = self._hand_colors[0]

        for idx, landmark in enumerate(landmarks):
            cx, cy = self._landmark_to_pixel(landmark, width, height)

            if idx == INDEX_FINGER_TIP:
                cv2.circle(frame, (cx, cy), 12, colors["index"], -1)
                cv2.circle(frame, (cx, cy), 12, (255, 255, 255), 2)
            elif idx == THUMB_TIP:
                cv2.circle(frame, (cx, cy), 10, colors["thumb"], -1)
            elif idx == WRIST:
                cv2.circle(frame, (cx, cy), 8, colors["wrist"], -1)
            else:
                cv2.circle(frame, (cx, cy), 5, colors["other"], -1)

    def _draw_connections(self, frame, landmarks, width: int, height: int,
                          colors: dict = None):
        """
        Draw lines between landmarks to form the hand skeleton.

        Sprint 11: Uses per-hand color for the skeleton lines.
        """
        if colors is None:
            colors = self._hand_colors[0]

        for connection in self.HAND_CONNECTIONS:
            start_lm = landmarks[connection.start]
            end_lm = landmarks[connection.end]

            start_pt = self._landmark_to_pixel(start_lm, width, height)
            end_pt = self._landmark_to_pixel(end_lm, width, height)

            cv2.line(frame, start_pt, end_pt, colors["skeleton"], 2)

    def get_landmark_position(self, frame, landmark_id: int):
        """
        Return pixel (x, y) of a specific landmark using the last inference result.

        HOW IT WORKS
        ------------
        detect() caches its result in self._last_results every frame.
        This method reads that cache — zero extra ML computation.

        WHY frame IS STILL A PARAMETER
        --------------------------------
        We need the frame dimensions (width, height) to convert normalized
        coordinates to pixels. We don't modify the frame here.

        Parameters
        ----------
        frame : np.ndarray
            Current webcam frame (used only to read .shape).
        landmark_id : int
            One of the 21 MediaPipe landmark IDs (0-20).
            Use the module-level constants: INDEX_FINGER_TIP, THUMB_TIP, WRIST.

        Returns
        -------
        tuple(int, int) | None
            Pixel (x, y) if a hand is detected, None otherwise.
            Callers MUST check for None before using the result.
        """
        if self._last_results is None:
            # No hand detected → reset smoother to avoid stale positions
            self._smoother.reset()
            return None
        if not self._last_results.hand_landmarks:
            self._smoother.reset()
            return None

        # Take the first detected hand (index 0).
        # Backward compatible: returns first hand's landmark.
        landmarks = self._last_results.hand_landmarks[0]

        h, w, _ = frame.shape
        landmark = landmarks[landmark_id]
        raw_x, raw_y = self._landmark_to_pixel(landmark, w, h)

        # ── Sprint 10: Apply EMA smoothing ────────────────────────────────
        smooth_x, smooth_y = self._smoother.update(landmark_id, raw_x, raw_y)
        return (smooth_x, smooth_y)

    # ────────────────────────────────────────────────────────────────────────
    # SPRINT 11: MULTI-HAND API
    # ────────────────────────────────────────────────────────────────────────

    def get_all_hands(self, frame) -> list:
        """
        Return smoothed landmark positions for ALL detected hands.

        This is the Sprint 11 multi-hand API. Unlike get_landmark_position()
        which returns data for only the first hand, this method returns data
        for every detected hand (up to max_hands=2).

        Each hand gets its own EMA smoother instance to prevent one hand's
        smoothing history from interfering with the other.

        Returns
        -------
        list[dict]   Each dict has:
            'index_tip' : (int, int) | None   Smoothed index finger position
            'thumb_tip' : (int, int) | None   Smoothed thumb position
            'hand_index': int                 0 or 1 (for color identification)
        """
        if self._last_results is None or not self._last_results.hand_landmarks:
            # No hands → clear all per-hand smoothers
            self._hand_smoothers.clear()
            return []

        h, w, _ = frame.shape
        hands = []

        for hand_idx, hand_landmarks in enumerate(self._last_results.hand_landmarks):
            # Create per-hand smoother on first detection
            if hand_idx not in self._hand_smoothers:
                self._hand_smoothers[hand_idx] = LandmarkSmoother(alpha=0.45)

            smoother = self._hand_smoothers[hand_idx]

            # Index finger tip
            idx_lm = hand_landmarks[INDEX_FINGER_TIP]
            raw_ix, raw_iy = self._landmark_to_pixel(idx_lm, w, h)
            smooth_ix, smooth_iy = smoother.update(INDEX_FINGER_TIP, raw_ix, raw_iy)

            # Thumb tip
            thumb_lm = hand_landmarks[THUMB_TIP]
            raw_tx, raw_ty = self._landmark_to_pixel(thumb_lm, w, h)
            smooth_tx, smooth_ty = smoother.update(THUMB_TIP, raw_tx, raw_ty)

            hands.append({
                "index_tip": (smooth_ix, smooth_iy),
                "thumb_tip": (smooth_tx, smooth_ty),
                "hand_index": hand_idx,
            })

        # Clean up smoothers for hands that disappeared
        active_indices = set(range(len(self._last_results.hand_landmarks)))
        stale = [k for k in self._hand_smoothers if k not in active_indices]
        for k in stale:
            del self._hand_smoothers[k]

        return hands