"""
keyboard.py — Sprint 4: Virtual Keyboard with Hover, Typing, and Word Suggestions

WHAT'S NEW IN SPRINT 4
-----------------------
Sprint 3: Hover detection, green highlight, pinch-to-type, text display
Sprint 4: Suggestion bar — 3 predicted word buttons above the text display
          select_suggestion() — auto-complete + replace partial word
          Layout upgraded to 1280×720 (larger, cleaner)

LAYOUT (1280 × 720 frame)
--------------------------
y=0-360:    Webcam feed (hand tracking)
y=365-405:  Suggestion bar  (3 word prediction boxes)
y=410-450:  Typed text display
y=455-505:  QWERTY row
y=512-562:  ASDFG row
y=569-619:  ZXCVB row
y=626-676:  SPACE / BACK row
"""

import cv2


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

        PAINTER'S ALGORITHM
        --------------------
        Named after physical oil painting: distant objects first, foreground last.
        Here:  background fill → border → letter label
        Each layer overwrites pixels beneath it.

        TEXT CENTERING MATH
        --------------------
        cv2.getTextSize() returns the bounding box of the text string.
        We use it to compute the pixel offset that centers text inside the key:

            text_x = key.x + (key.width  - text_width)  // 2
            text_y = key.y + (key.height + text_height) // 2

        Note: cv2 y-coordinates measure from TOP, but putText baseline
        is at the BOTTOM of the characters. That's why we ADD text_height
        for y but SUBTRACT text_width for x — they measure different things.
        """
        fill   = COLOR_KEY_HOVER   if hovered else COLOR_KEY_NORMAL
        border = COLOR_KEY_BORDER_HV if hovered else COLOR_KEY_BORDER
        tl     = (self.x,              self.y)
        br     = (self.x + self.width, self.y + self.height)

        cv2.rectangle(frame, tl, br, fill,   -1)    # fill
        cv2.rectangle(frame, tl, br, border,  2)    # border

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

        # Special keys (SPACE and BACK)
        sp_y = START_Y + 3 * (KEY_HEIGHT + GAP)
        sp_w = KEY_WIDTH * 5 + GAP * 4
        sp_x = START_X + (KEYBOARD_CANVAS_WIDTH - sp_w) // 2
        self.keys.append(Key("SPC",  sp_x,              sp_y, sp_w,          KEY_HEIGHT))
        self.keys.append(Key("BACK", sp_x + sp_w + GAP*2, sp_y, KEY_WIDTH*2, KEY_HEIGHT))

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

    def register_click(self):
        """
        Type the currently hovered key into typed_text.

        TYPED TEXT RULES
        -----------------
        'SPC'  → append space
        'BACK' → remove last character (s[:-1])
        other  → append letter
        """
        if self.hovered_key is None:
            return
        label = self.hovered_key.label
        if label == "SPC":
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

    def draw(self, frame, finger_pos=None):
        """
        Render the full keyboard UI: suggestions → text display → keys.

        DRAWING ORDER MATTERS
        ----------------------
        We draw in this order:
            1. Suggestion bar (topmost visual element)
            2. Text display bar
            3. Keyboard keys (bottommost)

        Painter's algorithm: bottom layers first if they overlap.
        Here they don't overlap but we still draw top-to-bottom for clarity.
        """
        hov_sug = self.get_hovered_suggestion(finger_pos)
        hov_key = self.get_hovered_key(finger_pos)

        self._draw_suggestion_bar(frame, hov_sug)
        self._draw_text_display(frame)

        for key in self.keys:
            key.draw(frame, hovered=(key is hov_key))

    def _draw_suggestion_bar(self, frame, hovered_box):
        """Render the 3 suggestion boxes side by side."""
        for box in self.suggestion_boxes:
            box.draw(frame, hovered=(box is hovered_box))

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