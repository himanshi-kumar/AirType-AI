"""
main.py — Sprint 8: Full AirType AI Pipeline with Premium UI Polish

WHAT CHANGED FROM SPRINT 7
---------------------------
Sprint 7: + SoundPlayer with synthesized audio cues (click, thud, blip, chime, chord)
Sprint 8: + Glassmorphism translucent key styling
          + Real-time typing stats bar (word count + WPM)
          + Visual audio pulse ring around SPEAK key during speech synthesis

PRIORITY ORDER IN PINCH HANDLING
----------------------------------
When a pinch fires, we check in this order:
    1. Is the finger over a SUGGESTION BOX? → auto-complete + play chime
    2. Is the finger over the SPEAK key?    → speak text + play chord
    3. Is the finger over SPC?              → auto-correct + append space + play thud
    4. Is the finger over BACK?             → delete char + play blip
    5. Is the finger over a letter?         → append letter + play click
"""

import time
import cv2
from hand_detector import HandDetector, INDEX_FINGER_TIP, THUMB_TIP
from keyboard import Keyboard
from gesture import PinchDetector
from prediction import WordPredictor
from speech import Speaker
from audio import SoundPlayer


# ── Resolution constants ──────────────────────────────────────────────────────
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720


# ── Component initialization ──────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

detector      = HandDetector()      # loads MediaPipe 7.5MB model once
keyboard      = Keyboard()          # creates all Key and SuggestionBox objects
pinch         = PinchDetector()     # stateful rising-edge + cooldown tracker
predictor     = WordPredictor()     # frequency dictionary + bigram table (instant, no files)
speaker       = Speaker()           # TTS engine — background thread, non-blocking
audio         = SoundPlayer()       # sound feedback — pre-synthesized, non-blocking
session_start = time.time()         # timestamp when typing session begins


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

    # ── Step 7: Draw keyboard + suggestions + UI overlays ─────────────────────
    # Keyboard.draw() handles all rendering:
    #   - Stats bar (word count, WPM)
    #   - Suggestion bar (3 blue boxes with predicted words)
    #   - Typed text display bar
    #   - All keyboard keys (glassmorphism translucent effect)
    #   - SPEAK pulsing ring when active
    keyboard.draw(
        frame,
        finger_pos=index_tip,
        is_speaking=speaker.is_speaking,
        session_start=session_start,
    )

    # ── Step 8: Pinch detection and action ────────────────────────────────────
    clicked = pinch.update(index_tip, thumb_tip)

    if clicked:
        # PRIORITY: suggestion first, SPEAK second, key third
        if keyboard.hovered_suggestion:
            keyboard.select_suggestion()
            audio.play_suggestion()                          # rising chime
        elif keyboard.hovered_key and keyboard.hovered_key.label == "SPEAK":
            # ── Sprint 6: Speak typed text aloud ───────────────────────
            speaker.speak(keyboard.typed_text)
            audio.play_speak()                               # activation chord
        elif keyboard.hovered_key:
            label = keyboard.hovered_key.label
            keyboard.register_click(predictor)
            # ── Sprint 7: Sound per key type ────────────────────────────
            if label == "SPC":
                audio.play_space()                           # deep thud
            elif label == "BACK":
                audio.play_backspace()                       # descending blip
            else:
                audio.play_keypress()                        # short click

    # ── Step 9: Display ───────────────────────────────────────────────────────
    cv2.imshow("AirType AI", frame)

    # ── Step 10: Quit ─────────────────────────────────────────────────────────
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ── Cleanup ───────────────────────────────────────────────────────────────
speaker.stop()
cap.release()
cv2.destroyAllWindows()