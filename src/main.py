"""
main.py — Sprint 3: Full AirType AI Pipeline

ARCHITECTURE: Orchestrator Pattern
------------------------------------
main.py does NOT contain any logic. It is a pure orchestrator:
  - Creates the components (camera, detector, keyboard, gesture)
  - Calls them in the correct order every frame
  - Passes data between them

WHY THIS MATTERS
----------------
If main.py contained collision detection logic, it would become a "God file"
that knows about everything and can't be tested in isolation. By keeping
each concern in its own module, we can:
  - Test HandDetector without a keyboard
  - Test PinchDetector with just coordinate tuples
  - Test Keyboard with simulated finger positions
  - Replace any component without touching the others

FRAME PIPELINE (per loop iteration)
-------------------------------------
1. Read raw BGR frame from webcam
2. Mirror it (so movement feels natural, like a mirror)
3. Run hand detection ML → draw skeleton on frame
4. Get index finger tip pixel position from cached result
5. Get thumb tip pixel position from cached result
6. Draw keyboard with current finger position (highlights hovered key)
7. Check if pinch fired this frame
8. If pinch fired AND a key is hovered → type the character
9. Display the frame
10. Check for quit key
"""

import cv2
from hand_detector import HandDetector, INDEX_FINGER_TIP, THUMB_TIP
from keyboard import Keyboard
from gesture import PinchDetector


# ── Component initialization ──────────────────────────────────────────────────
# Each component is created ONCE, before the loop.
# WHY ONCE? Creating a HandDetector loads a 7.5MB model into GPU/CPU memory.
# Creating it inside the loop would reload the model 30 times per second — unusable.

cap      = cv2.VideoCapture(0)      # 0 = default webcam
detector = HandDetector()           # loads MediaPipe model
keyboard = Keyboard()               # generates all key objects
pinch    = PinchDetector()          # stateful pinch tracker


# ── Main loop ─────────────────────────────────────────────────────────────────
# This loop runs at the webcam's frame rate (typically 30fps).
# Every iteration = one video frame processed.

while True:

    # ── Step 1: Read frame ────────────────────────────────────────────────────
    # cap.read() returns:
    #   ret   = True if a frame was successfully grabbed
    #   frame = the BGR NumPy array (shape: H × W × 3)
    # If ret is False, the webcam was disconnected or the video ended.
    ret, frame = cap.read()
    if not ret:
        break

    # ── Step 2: Mirror ────────────────────────────────────────────────────────
    # cv2.flip(frame, 1) flips horizontally.
    # WHY? Without mirroring, when you move your hand RIGHT,
    # the finger moves LEFT on screen — cognitively confusing.
    # Mirroring makes the interaction feel like looking in a mirror.
    frame = cv2.flip(frame, 1)

    # ── Step 3: Hand detection ────────────────────────────────────────────────
    # Runs ML inference, draws the skeleton, caches result internally.
    # After this call, detector._last_results holds the 21 landmark positions.
    frame = detector.detect(frame)

    # ── Step 4: Get finger positions ──────────────────────────────────────────
    # These read from the CACHED result (zero extra ML cost).
    # Returns None if no hand is detected — both keyboard and pinch handle None safely.
    index_tip = detector.get_landmark_position(frame, INDEX_FINGER_TIP)
    thumb_tip  = detector.get_landmark_position(frame, THUMB_TIP)

    # ── Step 5: Draw keyboard (with hover) ────────────────────────────────────
    # Passes index_tip so Keyboard can highlight the key under the finger.
    # Also draws the typed text display.
    keyboard.draw(frame, finger_pos=index_tip)

    # ── Step 6: Pinch detection + typing ─────────────────────────────────────
    # pinch.update() returns True ONLY on the frame the pinch starts.
    # If it returns True AND a key is hovered → register the keypress.
    clicked = pinch.update(index_tip, thumb_tip)
    if clicked:
        keyboard.register_click()

    # ── Step 7: Display ───────────────────────────────────────────────────────
    cv2.imshow("AirType AI", frame)

    # ── Step 8: Quit ──────────────────────────────────────────────────────────
    # waitKey(1) waits 1 millisecond, then checks for a key press.
    # & 0xFF masks to the lower 8 bits (needed on some 64-bit systems).
    # ord("q") = ASCII code for lowercase 'q' = 113.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ── Cleanup ───────────────────────────────────────────────────────────────────
# Always release the webcam and close windows.
# Without this, the webcam stays "in use" after the program exits —
# other apps (like FaceTime) can't access it until you reboot.
cap.release()
cv2.destroyAllWindows()