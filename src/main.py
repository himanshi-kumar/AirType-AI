"""
main.py — Sprint 12: Number/Symbol Keypad Toggle & Visual Ripple

WHAT CHANGED
------------
Sprint 8:  + Glassmorphism, stats bar, SPEAK pulse
Sprint 11: + Two-hand detection (each hand is an independent typing channel)
           + MultiPinchDetector for per-hand cooldown
           + FPS counter in stats bar
Sprint 12: + Keypad mode toggle (ABC <-> 123) with dual-tone chime
           + CLR (clear) key with descending reset chord
           + Dynamic ripple animations on every pinch click

MULTI-HAND ARCHITECTURE
-------------------------
1. detector.get_all_hands(frame) returns a list of hand dicts.
2. Each hand dict contains smoothed index_tip, thumb_tip, and hand_index.
3. MultiPinchDetector tracks rising-edge + cooldown per hand independently.
4. Keyboard.draw() highlights keys under ANY finger from any hand.
5. When either hand pinches a key, the corresponding action fires.

FPS COUNTER
-----------
Measures real loop throughput using a rolling window of frame timestamps.
Displayed in the stats bar (top-right) so users can monitor performance
impact of two-hand detection.
"""

import time
import cv2
from hand_detector import HandDetector, INDEX_FINGER_TIP, THUMB_TIP
from keyboard import Keyboard
from gesture import PinchDetector, MultiPinchDetector
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

detector      = HandDetector()          # loads MediaPipe model once (max_hands=2)
keyboard      = Keyboard()              # all Key + SuggestionBox objects
multi_pinch   = MultiPinchDetector()    # Sprint 11: per-hand pinch detection
predictor     = WordPredictor()         # BK-Tree + Bayesian autocorrect (Sprint 9)
speaker       = Speaker()              # TTS engine — background thread
audio         = SoundPlayer()          # synthesized audio cues
session_start = time.time()            # typing session start

# FPS tracking: rolling window of recent frame timestamps
_fps_timestamps = []
_current_fps = 0.0

# Backward-compatible: keep a single PinchDetector for fallback
pinch = PinchDetector()


# ── Main loop ─────────────────────────────────────────────────────────────────

while True:

    # ── Step 1: Capture ────────────────────────────────────────────────────────
    ret, frame = cap.read()
    if not ret:
        break

    # ── Step 2: Normalize resolution ──────────────────────────────────────────
    if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR)

    # ── Step 3: Mirror ────────────────────────────────────────────────────────
    frame = cv2.flip(frame, 1)

    # ── Step 4: Hand detection (ML inference) ─────────────────────────────────
    frame = detector.detect(frame)

    # ── Step 5: FPS measurement ───────────────────────────────────────────────
    # Rolling window: keep timestamps from the last 30 frames.
    # FPS = number_of_frames / time_span
    now = time.time()
    _fps_timestamps.append(now)
    if len(_fps_timestamps) > 30:
        _fps_timestamps.pop(0)
    if len(_fps_timestamps) >= 2:
        elapsed = _fps_timestamps[-1] - _fps_timestamps[0]
        if elapsed > 0:
            _current_fps = (len(_fps_timestamps) - 1) / elapsed

    # ── Step 6: Multi-hand processing (Sprint 11) ─────────────────────────────
    # Get all detected hands (up to 2) with smoothed landmark positions.
    hands = detector.get_all_hands(frame)

    # Collect all finger tip positions for keyboard drawing
    finger_positions = [h["index_tip"] for h in hands if h["index_tip"]]

    # Also maintain backward-compatible single-hand variables
    index_tip = hands[0]["index_tip"] if hands else None
    thumb_tip = hands[0]["thumb_tip"] if hands else None

    # ── Step 7: Word prediction ───────────────────────────────────────────────
    suggestions = predictor.get_suggestions(keyboard.typed_text)
    keyboard.set_suggestions(suggestions)

    # ── Step 8: Draw keyboard + suggestions + UI overlays ─────────────────────
    keyboard.draw(
        frame,
        finger_pos=index_tip,             # backward compat
        is_speaking=speaker.is_speaking,
        session_start=session_start,
        finger_positions=finger_positions, # Sprint 11: all fingers
        fps=_current_fps,                  # Sprint 11: FPS display
    )

    # ── Step 9: Multi-hand pinch detection and action ─────────────────────────
    # Check each hand independently for pinch events.
    active_hand_indices = set()

    for hand_data in hands:
        hand_idx = hand_data["hand_index"]
        active_hand_indices.add(hand_idx)

        clicked = multi_pinch.update(
            hand_idx,
            hand_data["index_tip"],
            hand_data["thumb_tip"],
        )

        if clicked:
            # Use this hand's index finger for hover detection
            this_finger = hand_data["index_tip"]

            # Check what this finger is hovering over
            hov_sug = keyboard.get_hovered_suggestion(this_finger)
            hov_key = keyboard.get_hovered_key(this_finger)

            # PRIORITY: suggestion first, SPEAK second, key third
            if hov_sug and hov_sug.word:
                keyboard.hovered_suggestion = hov_sug
                keyboard.select_suggestion()
                audio.play_suggestion()                          # rising chime
            elif hov_key and hov_key.label == "SPEAK":
                speaker.speak(keyboard.typed_text)
                audio.play_speak()                               # activation chord
            elif hov_key:
                label = hov_key.label
                keyboard.hovered_key = hov_key
                keyboard.register_click(predictor)
                # Sound per key type
                if label == "SPC":
                    audio.play_space()                           # deep thud
                elif label == "BACK":
                    audio.play_backspace()                       # descending blip
                elif label == "CLR":
                    audio.play_clear()                           # descending chord (Sprint 12)
                elif label in ("123", "ABC"):
                    audio.play_mode_switch()                     # dual-tone chime (Sprint 12)
                else:
                    audio.play_keypress()                        # short click

    # Tick cooldowns for hands that weren't detected this frame
    multi_pinch.tick_inactive(active_hand_indices)

    # ── Step 10: Display ──────────────────────────────────────────────────────
    cv2.imshow("AirType AI", frame)

    # ── Step 11: Quit ─────────────────────────────────────────────────────────
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ── Cleanup ───────────────────────────────────────────────────────────────
speaker.stop()
cap.release()
cv2.destroyAllWindows()