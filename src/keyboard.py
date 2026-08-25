"""
keyboard.py — Sprint 8: Premium UI Polish

WHAT'S NEW IN SPRINT 8
-----------------------
Sprint 6: SPEAK key (purple) triggers Text-to-Speech
Sprint 7: Sound feedback for every key event
Sprint 8: + Glassmorphism key fill (cv2.addWeighted translucent overlay)
          + SPEAK pulse animation (orange ring while TTS is active)
          + Stats bar (word count + WPM displayed above suggestions)

LAYOUT (1280 × 720 frame)
--------------------------
y=0-360:    Webcam feed (hand tracking)
y=330-362:  Stats bar  (word count | WPM)
y=365-405:  Suggestion bar  (3 word prediction boxes)
y=410-450:  Typed text display
y=455-505:  QWERTY row
y=512-562:  ASDFG row
y=569-619:  ZXCVB row
y=626-676:  SPACE / SPEAK / BACK row
"""

import cv2
import math
import time


# ── Layout constants (calibrated for 1280×720) ────────────────────────────────

KEY_WIDTH  = 58     # pixel width of one key
KEY_HEIGHT = 50     # pixel height of one key
GAP        = 8      # gap between adjacent keys

START_X = 50        # left margin of the keyboard canvas
START_Y = 455       # y-position of the top key row

KEYBOARD_CANVAS_WIDTH = 1180   # total width the keyboard occupies (1280 - 2×50)

# Suggestion bar geometry
SUGGESTION_BAR_Y1  = 365   # top edge of suggestion bar
SUGGESTION_BAR_Y2  = 405   # bottom edge of suggestion bar
SUGGESTION_PADDING = 10    # internal padding inside each suggestion box
SUGGESTION_GAP     = 12    # gap between suggestion boxes

# Text display bar geometry (sits between suggestion bar and keyboard)
TEXT_BAR_Y1 = 412
TEXT_BAR_Y2 = 450

# ── Colors (BGR) ──────────────────────────────────────────────────────────────
COLOR_KEY_NORMAL      = (35,  35,  35)    # very dark grey
COLOR_KEY_HOVER       = (0,  160,  60)    # green
COLOR_KEY_BORDER      = (120, 120, 120)   # grey border
COLOR_KEY_BORDER_HV   = (255, 255, 255)   # white border (hovered)
COLOR_TEXT_NORMAL     = (255, 255, 255)
COLOR_TEXT_HOVER      = (255, 255, 255)
COLOR_TYPED_TEXT      = (0,  255, 150)    # bright green for typed text
COLOR_SUGGESTION_BG   = (20,  20,  80)    # dark blue for suggestion box
COLOR_SUGGESTION_HV   = (60,  60, 180)    # lighter blue when hovered
COLOR_SUGGESTION_TEXT = (200, 220, 255)   # light blue text
COLOR_SUGGESTION_HV_T = (255, 255, 255)   # white text when hovered

# SPEAK key colors (purple to visually distinguish from typing keys)
COLOR_SPEAK_NORMAL    = (100, 40,  80)    # dark purple
COLOR_SPEAK_HOVER     = (180, 80, 160)    # bright purple when hovered
COLOR_SPEAK_BORDER    = (160, 100, 180)   # purple border

# Sprint 8 — glassmorphism overlay alpha and pulse colors
GLASS_ALPHA      = 0.55       # key fill opacity (0.0 = transparent, 1.0 = opaque)
COLOR_PULSE      = (0, 165, 255)   # orange in BGR — SPEAK pulse ring color
COLOR_STATS_BG   = (10,  10,  20)  # near-black stats bar background
COLOR_STATS_TEXT = (160, 220, 255) # light-blue stats text

# Stats bar geometry
STATS_BAR_Y1 = 330
STATS_BAR_Y2 = 362



class Key:
    """
    Represents one key on the virtual keyboard.

    STATELESS DRAWING
    -----------------
    Key does not store hover state — it's passed at draw time.
    This means Key objects never go "stale" between frames.
    Each frame = a fresh decision about what to draw.

    ATTRIBUTES
    ----------
    label  : str    Letter shown on the key
    x,y    : int    Top-left corner pixel position
    width  : int    Key width in pixels
    height : int    Key height in pixels
    """

    def __init__(self, label: str, x: int, y: int, width: int, height: int):
        self.label  = label
        self.x      = x
        self.y      = y
        self.width  = width
        self.height = height

    def draw(self, frame, hovered: bool = False):
        """
        Draw this key using the painter's algorithm: fill → border → text.

        GLASSMORPHISM (Sprint 8)
        ------------------------
        Instead of a solid filled rectangle, we composite a semi-transparent
        colored layer over the existing frame pixels using cv2.addWeighted().

        FORMULA
        --------
        dst = src1 * alpha + src2 * (1 - alpha) + gamma

        Where:
          src1  = colored fill rectangle (solid color)
          src2  = the existing frame pixels behind the key
          alpha = GLASS_ALPHA (0.55 = 55% fill, 45% see-through)
          gamma = 0 (no brightness offset)

        Result: key fill is semi-transparent, showing the webcam feed behind it.
        This creates a "frosted glass" depth effect.

        WHY cv2.addWeighted?
        --------------------
        It's a single optimized C++ call that blends two same-size NumPy arrays.
        Equivalent to: frame[roi] = fill * alpha + frame[roi] * (1 - alpha)
        but faster due to SIMD vectorization in OpenCV's native layer.
        """
        # SPEAK key uses distinct purple colors
        if self.label == "SPEAK":
            fill   = COLOR_SPEAK_HOVER  if hovered else COLOR_SPEAK_NORMAL
            border = COLOR_SPEAK_BORDER
        else:
            fill   = COLOR_KEY_HOVER   if hovered else COLOR_KEY_NORMAL
            border = COLOR_KEY_BORDER_HV if hovered else COLOR_KEY_BORDER

        tl = (self.x,              self.y)
        br = (self.x + self.width, self.y + self.height)

        # ── Glassmorphism fill (Sprint 8) ─────────────────────────────────
        # Extract the key's region of interest (ROI) from the current frame
        roi = frame[self.y : self.y + self.height, self.x : self.x + self.width]
        if roi.size > 0:
            import numpy as np
            # Create a solid-color overlay the same size as the ROI
            overlay = np.full_like(roi, fill[::-1] if len(fill) == 3 else fill)
            # Wait — fill is already BGR; np.full_like fills all channels equally.
            # We need to fill each channel separately:
            overlay[:, :, 0] = fill[0]   # Blue channel
            overlay[:, :, 1] = fill[1]   # Green channel
            overlay[:, :, 2] = fill[2]   # Red channel
            # Blend: GLASS_ALPHA * overlay + (1 - GLASS_ALPHA) * frame_roi
            blended = cv2.addWeighted(overlay, GLASS_ALPHA, roi, 1 - GLASS_ALPHA, 0)
            frame[self.y : self.y + self.height, self.x : self.x + self.width] = blended

        # Border (always fully opaque)
        cv2.rectangle(frame, tl, br, border, 2)

        # Centered text label
        font, fscale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2
        color = COLOR_TEXT_HOVER if hovered else COLOR_TEXT_NORMAL
        (tw, th), _ = cv2.getTextSize(self.label, font, fscale, thick)
        tx = self.x + (self.width  - tw) // 2
        ty = self.y + (self.height + th) // 2
        cv2.putText(frame, self.label, (tx, ty), font, fscale, color, thick)

    def contains(self, point: tuple) -> bool:
        """
        AABB collision: is pixel point inside this key's rectangle?

        A point (px, py) is inside if:
            x ≤ px ≤ x+width   AND   y ≤ py ≤ y+height

        This is called an Axis-Aligned Bounding Box test (AABB).
        "Axis-aligned" = edges are parallel to x and y axes (no rotation).
        Time complexity: O(1) — 4 comparisons regardless of anything.
        """
        if point is None:
            return False
        px, py = point
        return self.x <= px <= self.x + self.width and self.y <= py <= self.y + self.height


class SuggestionBox:
    """
    A single word-prediction suggestion shown above the keyboard.

    WHY A SEPARATE CLASS (not a Key)?
    -----------------------------------
    SuggestionBox and Key look visually similar but are fundamentally different:
    - Key has a fixed position computed at startup (layout is static)
    - SuggestionBox has a DYNAMIC label (changes every frame as user types)
    - SuggestionBox uses different colors to signal "this is a prediction, not a key"

    Using the same Key class would violate Open/Closed Principle:
    we'd have to add special cases everywhere ("if this is a suggestion key, ...").
    Separate class = separate responsibility.

    ATTRIBUTES
    ----------
    x, y     : int   Top-left corner
    width    : int   Box width
    height   : int   Box height (= SUGGESTION_BAR_Y2 - SUGGESTION_BAR_Y1)
    word     : str   The suggested word (changes every frame)
    """

    def __init__(self, x: int, y: int, width: int, height: int):
        self.x      = x
        self.y      = y
        self.width  = width
        self.height = height
        self.word   = ""    # set by Keyboard.set_suggestions() each frame

    def draw(self, frame, hovered: bool = False):
        """Draw this suggestion box. Empty word = invisible (skip drawing)."""
        if not self.word:
            return

        fill   = COLOR_SUGGESTION_HV  if hovered else COLOR_SUGGESTION_BG
        tl     = (self.x,              self.y)
        br     = (self.x + self.width, self.y + self.height)

        # Rounded-rectangle feel via filled rect + border
        cv2.rectangle(frame, tl, br, fill,         -1)
        cv2.rectangle(frame, tl, br, (80, 80, 200), 2)

        font, fscale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2
        color = COLOR_SUGGESTION_HV_T if hovered else COLOR_SUGGESTION_TEXT
        (tw, th), _ = cv2.getTextSize(self.word, font, fscale, thick)
        tx = self.x + (self.width  - tw) // 2
        ty = self.y + (self.height + th) // 2
        cv2.putText(frame, self.word, (tx, ty), font, fscale, color, thick)

    def contains(self, point: tuple) -> bool:
        """AABB collision test (same algorithm as Key.contains)."""
        if point is None or not self.word:
            return False
        px, py = point
        return self.x <= px <= self.x + self.width and self.y <= py <= self.y + self.height


class Keyboard:
    """
    Manages all keys, suggestion boxes, hover detection, and the text buffer.

    DATA FLOW (Sprint 4 addition)
    ------------------------------
    main.py calls predictor.get_suggestions(typed_text) → list of words
    main.py calls keyboard.set_suggestions(words) → updates suggestion boxes
    keyboard.draw() renders everything including the updated suggestions
    If pinch fires on a suggestion → keyboard.select_suggestion() auto-completes

    ATTRIBUTES
    ----------
    keys             : list[Key]            28 letter + special keys
    suggestion_boxes : list[SuggestionBox]  3 suggestion buttons
    typed_text       : str                  The full typed string
    hovered_key      : Key | None           Key under the finger (or None)
    hovered_suggestion : SuggestionBox | None  Suggestion box under finger (or None)
    """

    def __init__(self):
        self.keys: list            = []
        self.suggestion_boxes: list = []
        self.typed_text: str       = ""
        self.hovered_key           = None
        self.hovered_suggestion    = None
        self._create_keyboard()
        self._create_suggestion_bar()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _create_keyboard(self):
        """
        Programmatically generate all Key objects from the QWERTY row strings.

        CENTERING ALGORITHM
        --------------------
        row_pixel_width = n_keys * (KEY_WIDTH + GAP) - GAP
        row_start_x     = START_X + (KEYBOARD_CANVAS_WIDTH - row_pixel_width) // 2

        Example for QWERTYUIOP (10 keys):
            row_width = 10 × (58+8) - 8 = 652
            row_start_x = 50 + (1180 - 652) // 2 = 50 + 264 = 314
        """
        rows = ["QWERTYUIOP", "ASDFGHJKL", "ZXCVBNM"]

        for row_idx, row in enumerate(rows):
            y = START_Y + row_idx * (KEY_HEIGHT + GAP)
            row_w = len(row) * (KEY_WIDTH + GAP) - GAP
            x = START_X + (KEYBOARD_CANVAS_WIDTH - row_w) // 2
            for letter in row:
                self.keys.append(Key(letter, x, y, KEY_WIDTH, KEY_HEIGHT))
                x += KEY_WIDTH + GAP

        # Special keys (SPACE, SPEAK, and BACK)
        sp_y = START_Y + 3 * (KEY_HEIGHT + GAP)
        # Layout: [   SPC   ] [SPEAK] [ BACK ]
        # SPC is wide (5 keys), SPEAK and BACK are 2 keys each.
        spc_w   = KEY_WIDTH * 4 + GAP * 3                  # space bar width
        speak_w = KEY_WIDTH * 2 + GAP                       # SPEAK key width
        back_w  = KEY_WIDTH * 2 + GAP                       # BACK key width
        total_w = spc_w + GAP * 2 + speak_w + GAP * 2 + back_w
        sp_x    = START_X + (KEYBOARD_CANVAS_WIDTH - total_w) // 2

        self.keys.append(Key("SPC",   sp_x,                                sp_y, spc_w,   KEY_HEIGHT))
        self.keys.append(Key("SPEAK", sp_x + spc_w + GAP * 2,             sp_y, speak_w, KEY_HEIGHT))
        self.keys.append(Key("BACK",  sp_x + spc_w + GAP * 2 + speak_w + GAP * 2, sp_y, back_w,  KEY_HEIGHT))

    def _create_suggestion_bar(self, n: int = 3):
        """
        Create N evenly spaced SuggestionBox objects.

        WHY COMPUTE WIDTHS DYNAMICALLY?
        --------------------------------
        If we ever change n=3 to n=4 suggestions, only this method changes.
        Everything else adapts automatically. That's the value of computing
        layout from parameters instead of hardcoding pixel values.

        FORMULA
        --------
        total_gap   = (n-1) × SUGGESTION_GAP
        box_width   = (KEYBOARD_CANVAS_WIDTH - total_gap) / n
        box_x[i]    = START_X + i × (box_width + SUGGESTION_GAP)
        """
        total_gap = (n - 1) * SUGGESTION_GAP
        box_w = (KEYBOARD_CANVAS_WIDTH - total_gap) // n
        h = SUGGESTION_BAR_Y2 - SUGGESTION_BAR_Y1

        for i in range(n):
            x = START_X + i * (box_w + SUGGESTION_GAP)
            self.suggestion_boxes.append(
                SuggestionBox(x, SUGGESTION_BAR_Y1, box_w, h)
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def set_suggestions(self, words: list):
        """
        Update the words shown in the suggestion bar.

        Called by main.py every frame after predictor.get_suggestions().
        Words that don't fit in the bar are silently ignored.

        WHY SET (not return)?
        ----------------------
        The predictor computes suggestions; the keyboard DISPLAYS them.
        Passing the list in via set() keeps these responsibilities separate:
            predictor  → knows WHAT to suggest
            keyboard   → knows HOW to display it
        """
        for i, box in enumerate(self.suggestion_boxes):
            box.word = words[i] if i < len(words) else ""

    def get_hovered_key(self, finger_pos) -> "Key | None":
        """Linear search: which key contains the finger position? O(28)."""
        if finger_pos is None:
            self.hovered_key = None
            return None
        for key in self.keys:
            if key.contains(finger_pos):
                self.hovered_key = key
                return key
        self.hovered_key = None
        return None

    def get_hovered_suggestion(self, finger_pos) -> "SuggestionBox | None":
        """
        Check if the finger is over any suggestion box.

        WHY CHECK SUGGESTIONS SEPARATELY FROM KEYS?
        ---------------------------------------------
        Suggestion boxes sit in a different region of the frame (above the keyboard).
        We check them BEFORE keys so a pinch on a suggestion doesn't also fire
        a key below it. The caller (main.py) uses the result to decide which
        action to take.
        """
        if finger_pos is None:
            self.hovered_suggestion = None
            return None
        for box in self.suggestion_boxes:
            if box.contains(finger_pos):
                self.hovered_suggestion = box
                return box
        self.hovered_suggestion = None
        return None

    def register_click(self, predictor=None):
        """
        Type the currently hovered key into typed_text.

        TYPED TEXT RULES
        -----------------
        'SPC'  → append space (+ auto-correct the last word if misspelled)
        'BACK' → remove last character (s[:-1])
        other  → append letter

        AUTO-CORRECT ON SPACE (Sprint 5)
        ----------------------------------
        When the user presses SPACE, we check if the last word is misspelled.
        If it is, we replace it with the closest vocabulary match BEFORE
        appending the space. This matches phone keyboard behavior:
            User types:  "THW" + SPACE
            Result:      "THE "

        Parameters
        ----------
        predictor : WordPredictor | None
            The prediction engine. If provided and the last word is misspelled,
            the word is auto-corrected before the space is appended.
        """
        if self.hovered_key is None:
            return
        label = self.hovered_key.label
        if label == "SPC":
            # ── Auto-correct before appending space ───────────────────────
            if predictor is not None and self.typed_text:
                words = self.typed_text.split(" ")
                last_word = words[-1]
                if last_word:  # not empty (avoid double-space edge case)
                    correction = predictor.get_autocorrect(last_word)
                    if correction is not None:
                        words[-1] = correction
                        self.typed_text = " ".join(words)
            self.typed_text += " "
        elif label == "BACK":
            self.typed_text = self.typed_text[:-1]
        else:
            self.typed_text += label


    def select_suggestion(self):
        """
        Auto-complete typed_text with the currently hovered suggestion word.

        ALGORITHM: Replace Partial Word
        --------------------------------
        typed_text = "HAPP"
        suggestion = "HAPPY"

        1. Split typed_text by space: ["HAPP"]
        2. Drop the last element (the partial word): []
        3. Rejoin with spaces: ""
        4. Append suggestion + space: "HAPPY "

        typed_text = "HELLO WOR"
        suggestion = "WORLD"

        1. Split: ["HELLO", "WOR"]
        2. Drop last: ["HELLO"]
        3. Rejoin: "HELLO"
        4. Append: "HELLO WORLD "   ← note the trailing space (next word starts)

        WHY A TRAILING SPACE?
        ----------------------
        After auto-completing "WORLD", the user is starting the NEXT word.
        The space triggers next-word prediction in the predictor.
        This matches behavior of every phone keyboard.
        """
        if self.hovered_suggestion is None or not self.hovered_suggestion.word:
            return

        words = self.typed_text.split(" ")
        words = words[:-1]                    # drop the partial word
        words.append(self.hovered_suggestion.word)
        self.typed_text = " ".join(words) + " "   # rejoin + trailing space

    # ── Drawing ───────────────────────────────────────────────────────────────

    def draw(self, frame, finger_pos=None, is_speaking: bool = False,
              session_start: float = None):
        """
        Render the full keyboard UI: stats → suggestions → text display → keys.

        DRAWING ORDER MATTERS
        ----------------------
        1. Stats bar       (topmost visual element, Sprint 8)
        2. Suggestion bar
        3. Text display bar
        4. Keyboard keys
        5. SPEAK pulse ring (topmost overlay, drawn last, Sprint 8)

        Parameters
        ----------
        frame        : np.ndarray   Current video frame
        finger_pos   : tuple|None   Index finger tip pixel position
        is_speaking  : bool         True if TTS is currently active (Sprint 8)
        session_start: float|None   time.time() when typing began, for WPM (Sprint 8)
        """
        hov_sug = self.get_hovered_suggestion(finger_pos)
        hov_key = self.get_hovered_key(finger_pos)

        self._draw_stats_bar(frame, session_start)
        self._draw_suggestion_bar(frame, hov_sug)
        self._draw_text_display(frame)

        for key in self.keys:
            key.draw(frame, hovered=(key is hov_key))

        if is_speaking:
            self._draw_speak_pulse(frame)

    def _draw_suggestion_bar(self, frame, hovered_box):
        """Render the 3 suggestion boxes side by side."""
        for box in self.suggestion_boxes:
            box.draw(frame, hovered=(box is hovered_box))

    def _draw_stats_bar(self, frame, session_start: float = None):
        """
        Draw a slim stats bar showing word count and WPM.

        WORD COUNT
        ----------
        Count words by splitting on spaces and filtering empty strings.
        "HELLO WORLD " → ["HELLO", "WORLD"] → 2 words.

        WPM (Words Per Minute)
        ----------------------
        WPM = (words_typed / elapsed_minutes)
        Where elapsed_minutes = (now - session_start) / 60.
        Standard typing test measurement.

        We only show WPM after 10 seconds (avoids wild early spikes).
        """
        # Background
        cv2.rectangle(
            frame,
            (START_X, STATS_BAR_Y1),
            (START_X + KEYBOARD_CANVAS_WIDTH, STATS_BAR_Y2),
            COLOR_STATS_BG, -1
        )
        cv2.rectangle(
            frame,
            (START_X, STATS_BAR_Y1),
            (START_X + KEYBOARD_CANVAS_WIDTH, STATS_BAR_Y2),
            (40, 40, 60), 1
        )

        # Word count
        words = [w for w in self.typed_text.split(" ") if w]
        word_count = len(words)

        # WPM
        wpm_str = ""
        if session_start is not None:
            elapsed = time.time() - session_start
            if elapsed >= 10 and word_count > 0:
                wpm = int(word_count / (elapsed / 60))
                wpm_str = f"  |  {wpm} WPM"

        stats_text = f"Words: {word_count}{wpm_str}"

        cv2.putText(
            frame,
            stats_text,
            (START_X + 12, STATS_BAR_Y2 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            COLOR_STATS_TEXT,
            1,
        )

    def _draw_speak_pulse(self, frame):
        """
        Draw a pulsing orange ring around the SPEAK key while TTS is active.

        PULSE ANIMATION
        ---------------
        The ring radius oscillates using a sine wave driven by real time:
            radius = base_radius + amplitude * sin(2π * freq * t)

        This creates a smooth, continuous breathing/pulse effect at ~2Hz.
        The ring thickness also oscillates (2→3px) for extra visual weight.
        """
        speak_key = next((k for k in self.keys if k.label == "SPEAK"), None)
        if speak_key is None:
            return

        cx = speak_key.x + speak_key.width  // 2
        cy = speak_key.y + speak_key.height // 2
        base_r = max(speak_key.width, speak_key.height) // 2 + 6

        t = time.time()
        pulse     = math.sin(2 * math.pi * 2.0 * t)   # 2Hz pulse
        radius    = int(base_r + 6 * pulse)
        thickness = 2 if pulse > 0 else 3

        cv2.circle(frame, (cx, cy), radius, COLOR_PULSE, thickness)

    def _draw_text_display(self, frame):
        """
        Render the typed text in a dark bar between suggestions and keyboard.

        SLIDING WINDOW (last 40 chars)
        --------------------------------
        If typed_text is very long, only show the last 40 characters.
        This prevents text overflowing outside the bar.
        This is a sliding window: as you type, the window shifts right.
        """
        display_text = self.typed_text[-40:] + "|"

        # Dark background bar
        cv2.rectangle(frame,
                      (START_X, TEXT_BAR_Y1),
                      (START_X + KEYBOARD_CANVAS_WIDTH, TEXT_BAR_Y2),
                      (15, 15, 15), -1)
        cv2.rectangle(frame,
                      (START_X, TEXT_BAR_Y1),
                      (START_X + KEYBOARD_CANVAS_WIDTH, TEXT_BAR_Y2),
                      (70, 70, 70), 1)

        cv2.putText(
            frame,
            display_text,
            (START_X + 12, TEXT_BAR_Y2 - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            COLOR_TYPED_TEXT,
            2,
        )