"""
hand_detector.py — Sprint 2: Hand Detection using MediaPipe Tasks API

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
        max_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
    ):
        """
        Initialize the HandDetector — compile the model and configure it.

        WHY THESE PARAMETERS?
        ---------------------
        max_hands=1
            We only need one hand for keyboard typing. Detecting two costs
            more compute; keep it minimal.

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

            for hand_landmarks in results.hand_landmarks:
                # Draw skeleton lines FIRST so landmark dots appear on top
                self._draw_connections(frame, hand_landmarks, w, h)
                # Draw landmark dots
                self._draw_landmarks(frame, hand_landmarks, w, h)

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

    def _draw_landmarks(self, frame, landmarks, width: int, height: int):
        """
        Draw a colored circle at each of the 21 landmark positions.

        VISUAL DESIGN CHOICES
        ---------------------
        INDEX_FINGER_TIP (ID 8) → bright green, larger circle with white ring.
            This is the "pointer" for keyboard interaction. Make it obvious.

        THUMB_TIP (ID 4) → yellow.
            Will be used in Sprint 3 for pinch detection (index + thumb).

        WRIST (ID 0) → blue.
            Anchor/base of the hand skeleton.

        All others → small white dots.
            Knuckles and mid-segments. Keep them subtle.
        """
        for idx, landmark in enumerate(landmarks):
            cx, cy = self._landmark_to_pixel(landmark, width, height)

            if idx == INDEX_FINGER_TIP:
                cv2.circle(frame, (cx, cy), 12, (0, 255, 0), -1)       # green fill
                cv2.circle(frame, (cx, cy), 12, (255, 255, 255), 2)    # white ring
            elif idx == THUMB_TIP:
                cv2.circle(frame, (cx, cy), 10, (0, 255, 255), -1)     # yellow fill
            elif idx == WRIST:
                cv2.circle(frame, (cx, cy), 8, (255, 100, 0), -1)      # blue fill
            else:
                cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)    # white fill

    def _draw_connections(self, frame, landmarks, width: int, height: int):
        """
        Draw lines between landmarks to form the hand skeleton.

        HAND_CONNECTIONS is a set of Connection(start=int, end=int) objects.
        Each represents one bone/segment of the hand (thumb has 4 bones,
        each finger has 3, plus 4 palm connections = 21 total, 20 edges).

        We draw in semi-transparent cyan so the skeleton is visible but
        does not overpower the landmark dots drawn on top.
        """
        for connection in self.HAND_CONNECTIONS:
            start_lm = landmarks[connection.start]
            end_lm = landmarks[connection.end]

            start_pt = self._landmark_to_pixel(start_lm, width, height)
            end_pt = self._landmark_to_pixel(end_lm, width, height)

            cv2.line(frame, start_pt, end_pt, (0, 200, 200), 2)   # cyan line

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
            return None
        if not self._last_results.hand_landmarks:
            return None

        # Take the first detected hand (index 0).
        # We configured max_hands=1 so there's always at most one.
        landmarks = self._last_results.hand_landmarks[0]

        h, w, _ = frame.shape
        landmark = landmarks[landmark_id]
        return self._landmark_to_pixel(landmark, w, h)