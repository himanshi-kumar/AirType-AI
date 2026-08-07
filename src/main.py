"""
main.py — Sprint 4: Full AirType AI Pipeline with Word Prediction

WHAT CHANGED FROM SPRINT 3
---------------------------
Sprint 3: detect → get positions → draw keyboard → pinch → type
Sprint 4: + set camera to 1280×720 (better layout)
          + call predictor every frame to compute suggestions
          + pass suggestions to keyboard
          + pinch on suggestion → auto-complete word
          + pinch on key → type letter (unchanged)

PRIORITY ORDER IN PINCH HANDLING
----------------------------------
When a pinch fires, we check in this order:
    1. Is the finger over a SUGGESTION BOX? → auto-complete
    2. Is the finger over a KEY?            → type the letter

This priority ensures suggestions don't accidentally fire a key behind them.
"""

import cv2
from hand_detector import HandDetector, INDEX_FINGER_TIP, THUMB_TIP
from keyboard import Keyboard
from gesture import PinchDetector
from prediction import WordPredictor


# ── Resolution constants ──────────────────────────────────────────────────────
# Target frame size: 1280 × 720 (standard 720p HD).
# WHY 1280×720?
#   - Standard HD resolution — most webcams support it
#   - Wide enough to fit the full QWERTY keyboard (10 keys × 66px = 660px + margins)
#   - 720px tall: enough room for webcam feed (0–360) + UI (360–720)
#   - 16:9 aspect ratio = no distortion on widescreen monitors
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720


# ── Component initialization ──────────────────────────────────────────────────
cap = cv2.VideoCapture(0)

# Request 1280×720 from the webcam.
# set() returns True if the resolution is supported, False if not.
# If unsupported, OpenCV silently falls back to the nearest supported size.
# We then resize the frame manually (next step in the loop) to guarantee
# consistent dimensions regardless of what the webcam actually delivers.
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

detector  = HandDetector()      # loads MediaPipe 7.5MB model once
keyboard  = Keyboard()          # creates all Key and SuggestionBox objects
pinch     = PinchDetector()     # stateful rising-edge + cooldown tracker
predictor = WordPredictor()     # frequency dictionary + bigram table (instant, no files)


# ── Main loop ─────────────────────────────────────────────────────────────────

while True:

    # ── Step 1: Capture ────────────────────────────────────────────────────────
    ret, frame = cap.read()
    if not ret:
        break

    # ── Step 2: Normalize resolution ──────────────────────────────────────────
    # Even if the webcam can't deliver 1280×720, we resize the frame here.
    # All subsequent coordinate calculations assume FRAME_WIDTH × FRAME_HEIGHT.
    # Without this, landmark pixel positions wouldn't match key positions.
    #
    # cv2.resize interpolation:
    #   INTER_LINEAR = bilinear interpolation (good quality, fast)
    #   INTER_NEAREST = nearest-neighbour (fast, pixelated — avoid for display)
    if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR)

    # ── Step 3: Mirror ────────────────────────────────────────────────────────
    frame = cv2.flip(frame, 1)

    # ── Step 4: Hand detection (ML inference) ─────────────────────────────────
    frame = detector.detect(frame)

    # ── Step 5: Get finger positions from cached result ────────────────────────
    index_tip = detector.get_landmark_position(frame, INDEX_FINGER_TIP)
    thumb_tip  = detector.get_landmark_position(frame, THUMB_TIP)

    # ── Step 6: Word prediction ───────────────────────────────────────────────
    # Compute suggestions from the current typed text.
    # This runs every frame — it's a dictionary lookup, not ML inference.
    # Time cost: ~0.01ms. Negligible at 30fps.
    #
    # WHY EVERY FRAME?
    # Because typed_text changes whenever a pinch fires (previous step).
    # If we only updated predictions on pinch, the suggestion bar would
    # lag one frame behind. Updating every frame keeps it always current.
    suggestions = predictor.get_suggestions(keyboard.typed_text)
    keyboard.set_suggestions(suggestions)

    # ── Step 7: Draw keyboard + suggestions ───────────────────────────────────
    # Keyboard.draw() handles all rendering:
    #   - Suggestion bar (3 blue boxes with predicted words)
    #   - Typed text display bar
    #   - All 28 keyboard keys (green = hovered)
    keyboard.draw(frame, finger_pos=index_tip)

    # ── Step 8: Pinch detection and action ────────────────────────────────────
    clicked = pinch.update(index_tip, thumb_tip)

    if clicked:
        # PRIORITY: suggestion first, key second
        # If the finger is over a suggestion box, auto-complete the word.
        # Only if no suggestion is hovered do we check keyboard keys.
        if keyboard.hovered_suggestion:
            keyboard.select_suggestion()
        elif keyboard.hovered_key:
            keyboard.register_click()

    # ── Step 9: Display ───────────────────────────────────────────────────────
    cv2.imshow("AirType AI", frame)

    # ── Step 10: Quit ─────────────────────────────────────────────────────────
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ── Cleanup ───────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()