"""
keyboard.py — Sprint 3: Virtual Keyboard with Hover + Typing

WHAT'S NEW IN SPRINT 3
-----------------------
Sprint 1.2: Key class (draw a single key)
Sprint 1.3: Keyboard class (auto-generate all rows)
Sprint 3:   Hover detection (is a finger over a key?)
            Visual feedback (highlight the hovered key)
            Typing engine (register a keypress, build typed text)
            Backspace support
            Typed text display on screen

DESIGN PHILOSOPHY
-----------------
The Keyboard class does NOT know about hands or gestures.
It only answers two questions:
    1. "Which key is my finger currently over?" (get_hovered_key)
    2. "A click happened — type it!" (register_click)

The caller (main.py) bridges the gap:
    finger_pos → keyboard.get_hovered_key() → pinch? → keyboard.register_click()
"""

import cv2


# ── Visual constants ──────────────────────────────────────────────────────────

# Key dimensions in pixels
KEY_WIDTH = 60
KEY_HEIGHT = 60
GAP = 10                       # spacing between keys

# Start position: top-left corner of the keyboard on screen
START_X = 50
START_Y = 400                  # pushed to bottom of frame (frame is typically 480px tall)

# Total keyboard canvas width (used for centering rows)
KEYBOARD_CANVAS_WIDTH = 780

# Colors (BGR format — remember OpenCV is BGR, not RGB)
COLOR_KEY_NORMAL    = (40, 40, 40)       # dark grey fill
COLOR_KEY_HOVER     = (0, 180, 60)       # bright green fill when finger is over key
COLOR_KEY_BORDER    = (180, 180, 180)    # light grey border
COLOR_KEY_BORDER_HV = (255, 255, 255)    # white border when hovered
COLOR_TEXT_NORMAL   = (255, 255, 255)    # white text
COLOR_TEXT_HOVER    = (255, 255, 255)    # white text on hover
COLOR_TYPED_TEXT    = (0, 255, 150)      # green for the typed word display


class Key:
    """
    Represents a single keyboard key — its position, label, and how to draw itself.

    SINGLE RESPONSIBILITY
    ---------------------
    Key knows:
      - where it lives on screen (x, y, width, height)
      - what letter it represents (label)
      - how to draw itself in normal or hovered state

    Key does NOT know:
      - whether a finger is over it (that's Keyboard.get_hovered_key)
      - what happens when clicked (that's Keyboard.register_click)
      - anything about MediaPipe or gestures

    This separation means we can change the drawing style without touching
    any gesture or detection code, and vice versa.

    ATTRIBUTES
    ----------
    label  : str     The letter shown on the key (e.g. 'Q')
    x      : int     Left edge pixel position
    y      : int     Top edge pixel position
    width  : int     Key width in pixels
    height : int     Key height in pixels
    """

    def __init__(self, label: str, x: int, y: int, width: int, height: int):
        self.label = label
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def draw(self, frame, hovered: bool = False):
        """
        Draw this key onto the frame.

        WHY hovered IS A PARAMETER (not stored on self)
        ------------------------------------------------
        Hover state changes 30 times per second based on finger position.
        Storing it on the Key object would mean we mutate Key state every frame,
        which makes the code harder to reason about.
        Instead, we pass it as a parameter at draw time — Key stays stateless.
        Stateless objects are easier to test and debug.

        DRAWING ORDER (painter's algorithm)
        ------------------------------------
        1. Filled rectangle (background)
        2. Border rectangle (outline)
        3. Text label

        We draw background FIRST, then border on top, then text on top of that.
        This is the "painter's algorithm": paint background, then foreground.
        Same principle used in 3D rendering engines.

        Parameters
        ----------
        frame   : np.ndarray   The BGR frame to draw on (mutated in place).
        hovered : bool         True if the index finger is over this key.
        """
        fill_color   = COLOR_KEY_HOVER   if hovered else COLOR_KEY_NORMAL
        border_color = COLOR_KEY_BORDER_HV if hovered else COLOR_KEY_BORDER

        top_left     = (self.x, self.y)
        bottom_right = (self.x + self.width, self.y + self.height)

        # 1. Fill background
        cv2.rectangle(frame, top_left, bottom_right, fill_color, -1)

        # 2. Draw border (thickness=2, not filled)
        cv2.rectangle(frame, top_left, bottom_right, border_color, 2)

        # 3. Draw letter
        # Center the text inside the key.
        # cv2.getTextSize returns (width, height) of the text bounding box.
        # We use this to mathematically center the text — no guessing.
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        thickness  = 2
        text_color = COLOR_TEXT_HOVER if hovered else COLOR_TEXT_NORMAL

        (text_w, text_h), _ = cv2.getTextSize(self.label, font, font_scale, thickness)
        text_x = self.x + (self.width  - text_w) // 2
        text_y = self.y + (self.height + text_h) // 2

        cv2.putText(frame, self.label, (text_x, text_y), font, font_scale, text_color, thickness)

    def contains(self, point: tuple) -> bool:
        """
        Return True if a pixel point falls inside this key's bounding box.

        COLLISION DETECTION (AABB)
        ---------------------------
        AABB = Axis-Aligned Bounding Box.
        A key is a rectangle. A finger position is a point.
        "Is the point inside the rectangle?" is the collision test.

        Mathematically:
            point is inside if:
                key.x ≤ point.x ≤ key.x + key.width
            AND
                key.y ≤ point.y ≤ key.y + key.height

        WHY "AXIS-ALIGNED"?
        --------------------
        Our keys are always straight (not rotated). Axis-aligned means the
        rectangle's edges are parallel to the x and y axes. This allows the
        simple min/max check above. If keys were rotated, we'd need more
        complex math (SAT — Separating Axis Theorem).

        Parameters
        ----------
        point : (int, int)   Pixel (x, y) of the finger tip.

        Returns
        -------
        bool   True if the point is inside this key's rectangle.
        """
        px, py = point
        return (
            self.x <= px <= self.x + self.width
            and self.y <= py <= self.y + self.height
        )


class Keyboard:
    """
    Manages all keys, hover detection, and the typed text buffer.

    ATTRIBUTES
    ----------
    keys        : list[Key]   All 26 letter keys + SPACE + BACK.
    typed_text  : str         The string the user has typed so far.
    hovered_key : Key | None  The key currently under the finger (or None).
    """

    def __init__(self):
        self.keys: list = []
        self.typed_text: str = ""
        self.hovered_key = None
        self._create_keyboard()

    def _create_keyboard(self):
        """
        Auto-generate all key objects from the QWERTY layout.

        WHY NOT HARDCODE POSITIONS?
        ----------------------------
        26 keys × 4 parameters = 104 numbers to type manually and maintain.
        Instead, we define the LAYOUT (which letters, in which rows) and
        COMPUTE the positions algorithmically.

        CENTERING ALGORITHM
        --------------------
        Each row has a different number of keys (10, 9, 7).
        To center each row within the keyboard canvas:

            total_row_width = n_keys * key_width + (n_keys - 1) * gap
            x_start = canvas_left + (canvas_width - total_row_width) // 2

        Integer division (//) gives us a whole pixel — no fractional positions.
        """
        rows = [
            "QWERTYUIOP",   # 10 keys
            "ASDFGHJKL",    # 9 keys
            "ZXCVBNM",      # 7 keys
        ]

        for row_index, row in enumerate(rows):
            # Vertical position: each row is shifted down by (key_height + gap)
            y = START_Y + row_index * (KEY_HEIGHT + GAP)

            # Center this row horizontally
            row_pixel_width = len(row) * (KEY_WIDTH + GAP) - GAP
            x = START_X + (KEYBOARD_CANVAS_WIDTH - row_pixel_width) // 2

            for letter in row:
                self.keys.append(Key(letter, x, y, KEY_WIDTH, KEY_HEIGHT))
                x += KEY_WIDTH + GAP

        # ── Special keys ─────────────────────────────────────────────────────
        # SPACE bar: wider key, centered below the letter rows
        space_y   = START_Y + 3 * (KEY_HEIGHT + GAP)
        space_w   = KEY_WIDTH * 5 + GAP * 4   # same width as 5 normal keys
        space_x   = START_X + (KEYBOARD_CANVAS_WIDTH - space_w) // 2
        self.keys.append(Key("SPC", space_x, space_y, space_w, KEY_HEIGHT))

        # BACKSPACE: right of SPACE
        back_x = space_x + space_w + GAP * 2
        self.keys.append(Key("BACK", back_x, space_y, KEY_WIDTH * 2, KEY_HEIGHT))

    def get_hovered_key(self, finger_pos) -> "Key | None":
        """
        Find and return the key the finger is currently over.

        ALGORITHM: Linear search (O(n), n ≈ 28 keys)
        -----------------------------------------------
        We loop through every key and call key.contains(finger_pos).
        First match wins (keys don't overlap so there's at most one hit).

        WHY NOT A SPATIAL DATA STRUCTURE?
        -----------------------------------
        For 28 keys, linear search is perfectly fast — O(28) = constant time
        in practice. A spatial structure (like a quadtree or grid hash) would
        add complexity for zero measurable benefit at this scale.
        Rule: don't over-engineer. Optimize when profiling proves it's needed.

        Parameters
        ----------
        finger_pos : (int, int) | None
            Pixel position of index finger tip. None = no hand detected.

        Returns
        -------
        Key | None
            The hovered Key, or None if no key is under the finger.
        """
        if finger_pos is None:
            self.hovered_key = None
            return None

        for key in self.keys:
            if key.contains(finger_pos):
                self.hovered_key = key
                return key

        self.hovered_key = None
        return None

    def register_click(self):
        """
        Register a keypress on the currently hovered key.

        Called by main.py when a pinch gesture fires AND a key is hovered.
        The Keyboard does NOT detect pinches — it just responds to them.

        TYPED TEXT LOGIC
        ----------------
        'SPC'  → append a space character
        'BACK' → remove the last character (Python string slicing: s[:-1])
        other  → append the letter

        WHY s[:-1] FOR BACKSPACE?
        --------------------------
        In Python, s[:-1] means "everything except the last character".
        It's equivalent to s[:len(s)-1] but more idiomatic.
        It handles the empty string safely ('' [:-1] = '' — no crash).
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

    def draw(self, frame, finger_pos=None):
        """
        Draw the entire keyboard and the typed text display onto the frame.

        CHANGE FROM SPRINT 1
        ---------------------
        In Sprint 1, draw() drew all keys identically.
        Now it accepts finger_pos and passes the hover state to each key.
        The Keyboard decides which key is hovered; each Key decides how to draw itself.
        Responsibility is cleanly separated.

        Parameters
        ----------
        frame      : np.ndarray       BGR frame to draw on.
        finger_pos : (int, int) | None   Index finger tip pixel position.
        """
        # Determine which key (if any) is hovered this frame
        hovered = self.get_hovered_key(finger_pos)

        # Draw all keys, passing hovered=True only to the hovered key
        for key in self.keys:
            key.draw(frame, hovered=(key is hovered))

        # Draw the typed text display above the keyboard
        self._draw_text_display(frame)

    def _draw_text_display(self, frame):
        """
        Draw the typed text string in a box above the keyboard.

        DISPLAY LOGIC
        -------------
        - Show last 30 characters to prevent overflow (sliding window)
        - Show a blinking cursor (we simulate this with a static '|' for now)
        - Background box behind text for readability over webcam feed
        """
        display_text = self.typed_text[-30:] + "|"   # sliding window + cursor

        # Draw a dark semi-transparent background bar
        bar_y1 = START_Y - 60
        bar_y2 = START_Y - 10
        cv2.rectangle(frame, (START_X, bar_y1), (START_X + KEYBOARD_CANVAS_WIDTH, bar_y2),
                      (20, 20, 20), -1)
        cv2.rectangle(frame, (START_X, bar_y1), (START_X + KEYBOARD_CANVAS_WIDTH, bar_y2),
                      (80, 80, 80), 1)

        # Draw the text inside the bar
        cv2.putText(
            frame,
            display_text,
            (START_X + 10, bar_y2 - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            COLOR_TYPED_TEXT,
            2,
        )