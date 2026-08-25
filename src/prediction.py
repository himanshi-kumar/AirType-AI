"""
prediction.py — Sprint 4: Word Prediction Engine

WHAT IS WORD PREDICTION?
------------------------
Every time you type a partial word, the predictor suggests completions.
Example:
    You type "H"  → suggestions: ["HAVE", "HE", "HIS"]
    You type "HA" → suggestions: ["HAVE", "HAS", "HAD"]
    You type "HAV"→ suggestions: ["HAVE"]

This is called PREFIX-BASED COMPLETION — the same algorithm that powers
autocomplete in Google Search, smartphone keyboards, and IDEs.

THE ALGORITHM: Frequency-Ranked Prefix Matching
-------------------------------------------------
Step 1: We have a dictionary: { word → frequency_score }
        Higher score = more common word.
        "THE" has score 100. "ZEALOUS" has score 3.

Step 2: Given a prefix (e.g. "HA"):
        Filter the dictionary to keep only words that START WITH the prefix.
        Sort them by frequency (highest first).
        Return the top N.

Step 3: Time complexity:
        Naive: O(V) where V = vocabulary size (scan every word).
        Optimized: O(1) lookup using a Trie data structure.
        For our vocabulary of ~300 words, O(300) is instant.
        We'll learn Tries in Sprint 6 when vocabulary grows to 50,000 words.

WHAT IS N-GRAM PREDICTION?
---------------------------
N-gram = a sequence of N words.
A bigram = 2 words: ("I", "AM"), ("YOU", "ARE"), ("THE", "BEST")
A trigram = 3 words: ("I", "AM", "A"), ("ONCE", "UPON", "A")

Bigram language model:
    P(next_word | current_word) = count(current, next) / count(current)

Translation: "Given that the user just typed 'I',
              what word is most likely to come next?"

We store common bigrams as a lookup table:
    BIGRAMS["I"] = ["AM", "HAVE", "WILL", "CAN"]
    BIGRAMS["YOU"] = ["ARE", "HAVE", "CAN", "WILL"]

This is the foundation of GPT, BERT, and every language model —
they just have billions of parameters instead of our 300 hardcoded pairs.

ARCHITECTURE DECISION
----------------------
We use a static vocabulary (hardcoded frequency dict) because:
  - No internet required
  - No model file to download
  - Zero latency (dictionary lookup vs neural inference)
  - Perfect for a demo keyboard

In Sprint 7, we can replace this with a real language model (GPT-2 Tiny,
Llama-Nano, or a fine-tuned BERT) using the same WordPredictor interface.
The rest of the system won't need to change — that's good interface design.
"""


# ── QWERTY Key Center Coordinates (pixel positions) ──────────────────────────
# Derived from keyboard.py layout constants:
#   START_X = 50, START_Y = 455, KEY_WIDTH = 58, KEY_HEIGHT = 50, GAP = 8
#   KEYBOARD_CANVAS_WIDTH = 1180
#
# Each value is the (center_x, center_y) of the key in the 1280×720 frame.
# Row centering formula:
#   row_pixel_width = n_keys × (KEY_WIDTH + GAP) - GAP
#   row_start_x     = START_X + (KEYBOARD_CANVAS_WIDTH - row_pixel_width) // 2
#   center_x        = row_start_x + key_index × (KEY_WIDTH + GAP) + KEY_WIDTH // 2
#   center_y        = START_Y + row_index × (KEY_HEIGHT + GAP) + KEY_HEIGHT // 2
#
# WHY STORE PIXEL COORDINATES?
# We use Euclidean distance between key centers to weight substitution cost.
# Adjacent keys (W/E) have small pixel distance → low substitution cost.
# Distant keys (Q/M) have large pixel distance → high substitution cost.
# This makes the auto-corrector QWERTY-aware: it knows that typing 'W'
# instead of 'E' is a more likely typo than typing 'W' instead of 'M'.

KEY_CENTERS: dict = {
    # Row 0: QWERTYUIOP (10 keys, row_start_x = 314)
    "Q": (343, 480), "W": (409, 480), "E": (475, 480), "R": (541, 480),
    "T": (607, 480), "Y": (673, 480), "U": (739, 480), "I": (805, 480),
    "O": (871, 480), "P": (937, 480),
    # Row 1: ASDFGHJKL (9 keys, row_start_x = 347)
    "A": (376, 538), "S": (442, 538), "D": (508, 538), "F": (574, 538),
    "G": (640, 538), "H": (706, 538), "J": (772, 538), "K": (838, 538),
    "L": (904, 538),
    # Row 2: ZXCVBNM (7 keys, row_start_x = 413)
    "Z": (442, 596), "X": (508, 596), "C": (574, 596), "V": (640, 596),
    "B": (706, 596), "N": (772, 596), "M": (838, 596),
}


def get_substitution_cost(char1: str, char2: str) -> float:
    """
    Compute the cost of substituting char1 with char2.

    QWERTY-AWARE SUBSTITUTION
    --------------------------
    Standard Levenshtein treats all substitutions equally (cost = 1).
    But on a QWERTY keyboard, mistyping 'W' for 'E' (adjacent keys) is
    far more likely than mistyping 'W' for 'M' (opposite side of keyboard).

    We use the Euclidean distance between key centers to scale the cost:
        cost = 0.2 + 0.8 × min(2.0, pixel_distance / 90.0)

    - Adjacent keys (W↔E): ~66px apart → cost ≈ 0.79
    - Diagonal neighbors (Q↔S): ~110px → cost ≈ 1.18
    - Far-apart keys (Q↔M): ~500px → cost = 1.80 (capped)

    The formula:
    - 0.2 = minimum cost (even adjacent keys aren't free substitutions)
    - 0.8 × ... = scales from 0.0 to 1.6 based on distance
    - 90.0 = normalization constant (approximately one key-width + gap)
    - cap at 2.0 prevents extremely distant keys from dominating the score

    Parameters
    ----------
    char1 : str   Character in the input word.
    char2 : str   Character in the vocabulary word.

    Returns
    -------
    float   Substitution cost: 0.0 if identical, 0.2–1.8 based on key distance,
            1.0 if either character is not on the QWERTY layout.
    """
    if char1 == char2:
        return 0.0
    if char1 not in KEY_CENTERS or char2 not in KEY_CENTERS:
        return 1.0

    x1, y1 = KEY_CENTERS[char1]
    x2, y2 = KEY_CENTERS[char2]
    pixel_dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    return 0.2 + 0.8 * min(2.0, pixel_dist / 90.0)


def damerau_levenshtein_distance(s1: str, s2: str) -> float:
    """
    Compute the QWERTY-weighted Damerau-Levenshtein distance between two strings.

    WHAT IS DAMERAU-LEVENSHTEIN?
    -----------------------------
    Standard Levenshtein distance counts the minimum edits (insertions,
    deletions, substitutions) to transform s1 into s2.

    Damerau-Levenshtein adds a fourth operation: TRANSPOSITION — swapping
    two adjacent characters. This is critical for keyboards because
    transposition is one of the most common typing errors:
        "HPAY" → "HAPY" → "HAPPY"

    Without transposition support, "HPAP" → "HAPPY" costs 3 edits.
    With transposition, it costs 2 (swap H↔P, then insert Y) — much closer
    to the human's intended input.

    QWERTY WEIGHTING
    -----------------
    Substitution costs are NOT uniform — they depend on the physical distance
    between keys on the QWERTY layout (see get_substitution_cost).

    FIRST-LETTER PENALTY
    ----------------------
    Users rarely mistype the first letter of a word. If the first letters
    differ, we add a 1.5 penalty to the total distance. This prevents
    short unrelated words from polluting the suggestions.
    Example: "COMPUTW" should match "COMPUTER" (same first letter),
             not "OUT" (different first letter, lower raw distance).

    DP RECURRENCE
    --------------
    dp[i][j] = minimum cost to transform s1[0..i-1] into s2[0..j-1]

    dp[i][j] = min(
        dp[i-1][j]   + 1.0,                    # delete s1[i-1]
        dp[i][j-1]   + 1.0,                    # insert s2[j-1]
        dp[i-1][j-1] + sub_cost(s1[i-1], s2[j-1]),  # substitute
        dp[i-2][j-2] + 1.0  (if s1[i-1]==s2[j-2] and s1[i-2]==s2[j-1])  # transpose
    )

    TIME COMPLEXITY: O(m × n) where m, n = lengths of s1, s2.
    For our vocabulary of ~300 words with average length 5:
        300 × 5 × 7 = ~10,500 operations per query. Instant at 30fps.

    Parameters
    ----------
    s1 : str   Input word (what the user typed).
    s2 : str   Candidate word from vocabulary.

    Returns
    -------
    float   Weighted edit distance (lower = more similar).
    """
    m, n = len(s1), len(s2)

    # Build DP table
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]

    # Base cases: transforming empty string to/from a prefix
    for i in range(m + 1):
        dp[i][0] = float(i)
    for j in range(n + 1):
        dp[0][j] = float(j)

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Substitution cost: 0 if chars match, QWERTY-weighted otherwise
            if s1[i - 1] == s2[j - 1]:
                sub_cost = 0.0
            else:
                sub_cost = get_substitution_cost(s1[i - 1], s2[j - 1])

            dp[i][j] = min(
                dp[i - 1][j] + 1.0,            # deletion
                dp[i][j - 1] + 1.0,            # insertion
                dp[i - 1][j - 1] + sub_cost,   # substitution
            )

            # Transposition: swap two adjacent characters
            # Only valid when both chars exist and form a swap pattern
            if (i > 1 and j > 1
                    and s1[i - 1] == s2[j - 2]
                    and s1[i - 2] == s2[j - 1]):
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + 1.0)

    distance = dp[m][n]

    # First-letter mismatch penalty
    if m > 0 and n > 0 and s1[0] != s2[0]:
        distance += 1.5

    return distance


# ── Vocabulary: Top 300 English words with relative frequency scores ──────────
# Score = relative frequency (higher = more common).
# Based on the Oxford English Corpus frequency list.
#
# WHY UPPERCASE?
# All our keyboard input is uppercase (QWERTY keys). We keep vocabulary
# uppercase to match — no case conversion needed at runtime.

WORD_FREQUENCIES: dict = {
    # Function words (ultra-high frequency)
    "THE": 100, "BE": 97, "TO": 96, "OF": 95, "AND": 94,
    "A": 93, "IN": 92, "THAT": 89, "HAVE": 88, "I": 87,
    "IT": 86, "FOR": 85, "NOT": 84, "ON": 83, "WITH": 82,
    "HE": 81, "AS": 80, "YOU": 79, "DO": 78, "AT": 77,
    "THIS": 76, "BUT": 75, "HIS": 74, "BY": 73, "FROM": 72,
    "THEY": 71, "WE": 70, "SAY": 69, "HER": 68, "SHE": 67,
    "OR": 66, "AN": 65, "WILL": 64, "MY": 63, "ONE": 62,
    "ALL": 61, "WOULD": 60, "THERE": 59, "THEIR": 58, "WHAT": 57,

    # Common verbs
    "GO": 56, "COME": 55, "GET": 54, "MAKE": 53, "KNOW": 52,
    "THINK": 51, "TAKE": 50, "SEE": 49, "LOOK": 48, "WANT": 47,
    "GIVE": 46, "USE": 45, "FIND": 44, "TELL": 43, "ASK": 42,
    "WORK": 41, "SEEM": 40, "FEEL": 39, "TRY": 38, "LEAVE": 37,
    "CALL": 36, "KEEP": 35, "LET": 34, "BEGIN": 33, "SHOW": 32,
    "HEAR": 31, "PLAY": 30, "RUN": 29, "MOVE": 28, "LIVE": 27,
    "BELIEVE": 26, "HOLD": 25, "BRING": 24, "HAPPEN": 23, "WRITE": 22,
    "PROVIDE": 21, "STAND": 20, "LOSE": 19, "PAY": 18, "MEET": 17,
    "INCLUDE": 16, "CONTINUE": 15, "SET": 14, "LEARN": 13, "CHANGE": 12,
    "NEED": 45, "HELP": 40, "START": 35, "LOVE": 33, "LIKE": 50,

    # Common nouns
    "TIME": 55, "YEAR": 54, "PEOPLE": 53, "WAY": 52, "DAY": 51,
    "MAN": 50, "WOMAN": 49, "CHILD": 48, "WORLD": 47, "LIFE": 46,
    "HAND": 45, "PART": 44, "PLACE": 43, "CASE": 42, "WEEK": 41,
    "COMPANY": 40, "SYSTEM": 39, "PROGRAM": 38, "QUESTION": 37, "WORK": 36,
    "GOVERNMENT": 35, "NUMBER": 34, "NIGHT": 33, "POINT": 32, "HOME": 31,
    "WATER": 30, "ROOM": 29, "MOTHER": 28, "AREA": 27, "MONEY": 26,
    "STORY": 25, "FACT": 24, "MONTH": 23, "LOT": 22, "RIGHT": 21,
    "STUDY": 20, "BOOK": 19, "EYE": 18, "JOB": 17, "WORD": 16,
    "BUSINESS": 15, "ISSUE": 14, "SIDE": 13, "KIND": 12, "HEAD": 11,
    "HOUSE": 30, "SCHOOL": 35, "STATE": 32, "FAMILY": 38, "STUDENT": 25,
    "GROUP": 28, "COUNTRY": 30, "PROBLEM": 27, "HAND": 40, "GAME": 22,
    "IDEA": 20, "BODY": 18, "INFORMATION": 15, "BACK": 35, "PARENT": 12,
    "FACE": 25, "OTHERS": 20, "LEVEL": 18, "OFFICE": 15, "DOOR": 14,
    "HEALTH": 18, "PERSON": 30, "ART": 16, "WAR": 20, "HISTORY": 15,
    "PARTY": 18, "RESULT": 20, "CHANGE": 22, "MORNING": 15, "REASON": 18,
    "RESEARCH": 12, "GIRL": 20, "GUY": 18, "MOMENT": 15, "AIR": 16,
    "TEACHER": 14, "FORCE": 18, "EDUCATION": 12, "NEVER": 25, "ALWAYS": 22,

    # Common adjectives
    "GOOD": 55, "NEW": 53, "FIRST": 51, "LAST": 49, "LONG": 47,
    "GREAT": 45, "LITTLE": 43, "OWN": 41, "OLD": 39, "RIGHT": 37,
    "BIG": 35, "HIGH": 33, "SMALL": 31, "LARGE": 29, "NEXT": 27,
    "EARLY": 25, "YOUNG": 23, "IMPORTANT": 21, "PUBLIC": 19, "BAD": 17,
    "SAME": 30, "ABLE": 25, "FREE": 22, "REAL": 20, "BEST": 35,
    "BLACK": 18, "WHITE": 18, "SURE": 22, "TRUE": 20, "WHOLE": 16,
    "HAPPY": 18, "HARD": 22, "OPEN": 20, "POSSIBLE": 15, "FULL": 18,
    "SPECIAL": 15, "EASY": 20, "CLEAR": 18, "RECENT": 12, "MAIN": 15,
    "SOCIAL": 14, "SIMPLE": 16, "HUMAN": 18, "STRONG": 16, "LIGHT": 18,
    "BEAUTIFUL": 12, "AMAZING": 14, "DIFFERENT": 18, "CURRENT": 14, "NATURAL": 12,

    # Common adverbs/prepositions/conjunctions
    "UP": 45, "OUT": 43, "ABOUT": 41, "THEN": 39, "WHEN": 37,
    "NOW": 35, "ALSO": 33, "JUST": 31, "BECAUSE": 29, "AFTER": 27,
    "BEFORE": 25, "HERE": 23, "HOW": 21, "VERY": 19, "MOST": 17,
    "EVEN": 30, "SO": 45, "WELL": 28, "BACK": 26, "MORE": 35,
    "THAN": 30, "ONLY": 28, "BOTH": 20, "ANY": 30, "STILL": 22,
    "OVER": 22, "NEVER": 25, "DOWN": 25, "AWAY": 18, "AGAIN": 22,
    "OFF": 20, "AROUND": 18, "MAYBE": 16, "ALREADY": 18, "TOGETHER": 15,
    "OFTEN": 16, "HOWEVER": 14, "REALLY": 20, "SOMETHING": 22, "SOMEONE": 18,
    "EVERYTHING": 14, "NOTHING": 16, "ANYTHING": 14, "EVERYONE": 12, "SOMEWHERE": 10,
    "HELLO": 20, "HI": 18, "THANKS": 16, "PLEASE": 18, "SORRY": 15,
    "YES": 22, "NO": 25, "OK": 20, "OKAY": 18, "WOW": 12,

    # Technology / AI (relevant to AirType's domain)
    "COMPUTER": 18, "CODE": 16, "DATA": 20, "MODEL": 18, "NETWORK": 14,
    "PYTHON": 14, "ALGORITHM": 12, "FUNCTION": 14, "CLASS": 13, "OBJECT": 12,
    "MACHINE": 14, "LEARNING": 16, "ARTIFICIAL": 12, "INTELLIGENCE": 12,
    "KEYBOARD": 16, "CAMERA": 14, "SCREEN": 16, "FINGER": 14, "HAND": 22,
    "GESTURE": 14, "DETECTION": 12, "TRACKING": 12, "VISION": 14, "IMAGE": 16,
}


# ── Bigram table: common next-word suggestions ────────────────────────────────
# BIGRAMS[word_A] = [word_B1, word_B2, ...] means:
#     "After typing word_A, these are the most likely next words."
#
# This is a first-order Markov chain: P(next | current) — the next word
# depends ONLY on the current word, not the entire history.
# GPT models are higher-order: they consider the full sentence history.

BIGRAMS: dict = {
    "I": ["AM", "HAVE", "WILL", "THINK", "LOVE", "NEED", "WANT", "KNOW"],
    "YOU": ["ARE", "HAVE", "CAN", "WILL", "NEED", "KNOW", "SHOULD", "WANT"],
    "WE": ["ARE", "HAVE", "NEED", "CAN", "WILL", "SHOULD", "WANT", "KNOW"],
    "THEY": ["ARE", "HAVE", "WILL", "CAN", "NEED", "WANT", "SHOULD", "KNOW"],
    "HE": ["IS", "HAS", "WILL", "CAN", "SAID", "WAS", "WANTS", "KNOWS"],
    "SHE": ["IS", "HAS", "WILL", "CAN", "SAID", "WAS", "WANTS", "KNOWS"],
    "IT": ["IS", "HAS", "WAS", "CAN", "WILL", "SEEMS", "LOOKS", "MAKES"],
    "THE": ["BEST", "SAME", "FIRST", "MOST", "NEW", "LAST", "NEXT", "BIG"],
    "A": ["NEW", "GOOD", "GREAT", "BIG", "SMALL", "LOT", "FEW", "PART"],
    "THIS": ["IS", "WAS", "CAN", "WILL", "MEANS", "MAKES", "SHOWS", "HELPS"],
    "MY": ["NAME", "LIFE", "WORK", "FAMILY", "FRIEND", "HAND", "GOAL", "IDEA"],
    "YOUR": ["NAME", "LIFE", "WORK", "FAMILY", "FRIEND", "HAND", "GOAL", "IDEA"],
    "IS": ["THE", "A", "VERY", "NOT", "GOOD", "GREAT", "IMPORTANT", "EASY"],
    "ARE": ["THE", "A", "VERY", "NOT", "GOOD", "GREAT", "IMPORTANT", "YOU"],
    "HAVE": ["A", "THE", "TO", "BEEN", "SOME", "MANY", "NO", "GOOD"],
    "DO": ["NOT", "YOU", "WE", "I", "THEY", "THIS", "THE", "IT"],
    "NOT": ["THE", "A", "ONLY", "JUST", "ALWAYS", "SURE", "GOOD", "EASY"],
    "HELLO": ["WORLD", "THERE", "EVERYONE", "FRIEND", "TEAM", "ALL"],
    "THANK": ["YOU", "GOD"],
    "THANKS": ["FOR", "TO", "A"],
    "GOOD": ["MORNING", "NIGHT", "LUCK", "JOB", "WORK", "IDEA", "TIME", "DAY"],
    "MACHINE": ["LEARNING", "VISION", "CODE", "INTELLIGENCE", "DATA", "MODEL"],
    "ARTIFICIAL": ["INTELLIGENCE", "NEURAL", "LEARNING"],
    "COMPUTER": ["VISION", "SCIENCE", "LEARNING", "CODE", "PROGRAM", "SYSTEM"],
}


class WordPredictor:
    """
    Predicts word completions and next-word suggestions.

    SINGLE RESPONSIBILITY
    ---------------------
    WordPredictor answers one question: "Given what the user has typed,
    what words should we suggest?"

    It does NOT know about fingers, keyboards, or frames.
    It only operates on strings.

    TWO MODES
    ----------
    1. PREFIX COMPLETION: User is mid-word → complete it.
       Input:  typed_text = "HAPP"
       Output: ["HAPPY", "HAPPEN"]

    2. NEXT-WORD PREDICTION: User just finished a word (last char = space) → suggest the next word.
       Input:  typed_text = "GOOD "
       Output: ["MORNING", "NIGHT", "LUCK"]

    ATTRIBUTES
    ----------
    vocabulary : dict[str, int]
        Word → frequency score. Higher score = higher priority in suggestions.
    bigrams : dict[str, list[str]]
        Word → list of likely next words.
    """

    def __init__(
        self,
        vocabulary: dict = None,
        bigrams: dict = None,
    ):
        # Use the built-in vocabulary by default.
        # Injecting them as parameters makes the class testable with custom data.
        self.vocabulary = vocabulary or WORD_FREQUENCIES
        self.bigrams = bigrams or BIGRAMS

    def get_suggestions(self, typed_text: str, n: int = 3) -> list:
        """
        Main entry point. Analyze typed_text and return N suggestions.

        ALGORITHM
        ----------
        1. Parse the typed_text to extract the "current word" (partial or full).
        2. If the last character is a space → next-word prediction mode.
        3. Otherwise → prefix completion mode.

        PARSING LOGIC
        --------------
        typed_text = "HELLO WORLD WO"
        words = ["HELLO", "WORLD", "WO"]
        current_partial = "WO"       (user is mid-word)
        last_complete   = "WORLD"    (the word before)

        typed_text = "HELLO WORLD "
        words = ["HELLO", "WORLD", ""]
        current_partial = ""         (just finished a word)
        last_complete   = "WORLD"    (predict what comes after WORLD)

        Parameters
        ----------
        typed_text : str   Everything the user has typed so far (uppercase).
        n          : int   Number of suggestions to return. Default 3.

        Returns
        -------
        list[str]   Up to n suggestions. Empty list if nothing sensible to suggest.
        """
        if not typed_text.strip():
            # Nothing typed yet — show most common words
            return self._top_n_words(n)

        words = typed_text.split(" ")
        current_partial = words[-1]          # what the user is currently typing
        last_complete   = words[-2] if len(words) >= 2 else ""  # previous full word

        if current_partial == "":
            # User just pressed SPACE → predict the NEXT word
            return self._predict_next(last_complete, n)
        else:
            # User is mid-word → complete the prefix
            return self._predict_completions(current_partial, n)

    def _predict_completions(self, prefix: str, n: int) -> list:
        """
        Return top N words from vocabulary that START WITH the prefix.
        Falls back to FUZZY MATCHING (Damerau-Levenshtein) if prefix yields < N.

        ALGORITHM: Filter + Sort + Fuzzy Backoff
        ------------------------------------------
        1. Filter: keep only words where word.startswith(prefix)
        2. Sort: by frequency score (descending — highest first)
        3. Slice: keep only the top N
        4. NEW — If fewer than N prefix matches exist, fill remaining slots
           with the closest Damerau-Levenshtein matches from the vocabulary.
           This means typos like "THW" still produce useful suggestions ("THE").

        TIME COMPLEXITY
        ---------------
        Prefix phase: O(V log V) where V = vocabulary size (~300 words).
        Fuzzy phase:  O(V × m × k) where m = prefix length, k = avg word length.
        Total: ~10,000 operations at worst. Still instant at 30fps.

        Parameters
        ----------
        prefix : str   Uppercase partial word (e.g. "HA").
        n      : int   Max number of results.

        Returns
        -------
        list[str]   Matching words sorted by frequency, up to n entries.
        """
        # Don't suggest if prefix is empty
        if len(prefix) < 1:
            return []

        # ── Phase 1: Exact prefix matches ─────────────────────────────────
        # Filter: keep only words that start with the prefix
        matches = [
            (word, score)
            for word, score in self.vocabulary.items()
            if word.startswith(prefix) and word != prefix  # don't suggest exact match
        ]

        # Sort by score, descending (highest frequency first)
        # key=lambda item: item[1] → sort by the score (second element of tuple)
        matches.sort(key=lambda item: item[1], reverse=True)

        # Return only the word labels (not the scores)
        suggestions = [word for word, _ in matches[:n]]

        # ── Phase 2: Fuzzy backoff (Damerau-Levenshtein) ──────────────────
        # If prefix matching didn't fill all N slots, find close matches.
        # Example: user typed "THW" — no exact prefix match → fuzzy finds "THE".
        if len(suggestions) < n:
            # Maximum acceptable edit distance scales with input length.
            # Short words (2-3 chars) allow small edits; longer words allow more.
            max_allowed = max(1.5, len(prefix) * 0.8)

            candidates = []
            for word in self.vocabulary:
                if word in suggestions:
                    continue  # already in the list from prefix matching
                dist = damerau_levenshtein_distance(prefix, word)
                if dist <= max_allowed:
                    candidates.append((word, dist, self.vocabulary[word]))

            # Sort by (distance ASC, frequency DESC) — closest first,
            # break ties by popularity.
            candidates.sort(key=lambda x: (x[1], -x[2]))

            for word, _, _ in candidates:
                if word not in suggestions:
                    suggestions.append(word)
                    if len(suggestions) == n:
                        break

        return suggestions

    def _predict_next(self, last_word: str, n: int) -> list:
        """
        Predict the most likely next word after last_word.

        Uses the BIGRAMS lookup table (first-order Markov model).
        Falls back to top-N most frequent words if no bigram exists.

        WHY FALL BACK TO TOP-N?
        -------------------------
        Our bigram table is small (30 entries). If the user typed "ZEBRA",
        there's no bigram for it. Rather than returning nothing,
        return the most universally common words: ["THE", "A", "AND"].
        This is called "backoff" — a standard NLP technique.
        Production systems use Kneser-Ney smoothing, but backoff is equivalent
        for our scale.

        Parameters
        ----------
        last_word : str   The word the user just finished typing.
        n         : int   Max number of results.

        Returns
        -------
        list[str]   Up to n predicted next words.
        """
        if last_word in self.bigrams:
            # Return top N from the bigram list
            return self.bigrams[last_word][:n]

        # Backoff: return top N most frequent words globally
        return self._top_n_words(n)

    def _top_n_words(self, n: int) -> list:
        """
        Return the N most frequent words in the vocabulary.

        Used as a fallback when no prefix or bigram match exists.
        "THE", "BE", "TO" are always useful defaults.
        """
        sorted_words = sorted(
            self.vocabulary.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        return [word for word, _ in sorted_words[:n]]

    def get_autocorrect(self, word: str) -> str:
        """
        Return the best spelling correction for a misspelled word.

        WHEN IS THIS CALLED?
        ----------------------
        When the user presses SPACE, the keyboard extracts the last word
        and asks: "Is this word misspelled? If so, what should it be?"

        ALGORITHM
        ----------
        1. If the word already exists in the vocabulary → return None (no fix needed).
        2. Compute Damerau-Levenshtein distance to every vocabulary word.
        3. Pick the word with the lowest distance.
        4. If the best distance is within a threshold → return correction.
        5. Otherwise → return None (word is too far from anything we know).

        THRESHOLD FORMULA
        ------------------
        max_allowed = max(1.5, len(word) * 0.8)

        - Short words ("HT", 2 chars): threshold = 1.6 → only very close matches
        - Medium words ("HAPY", 4 chars): threshold = 3.2 → more tolerance
        - Long words ("COMPUTW", 7 chars): threshold = 5.6 → even more tolerance

        This scales linearly with word length because longer words naturally
        accumulate more typos during air-typing.

        Parameters
        ----------
        word : str   The word to check (uppercase).

        Returns
        -------
        str | None   Corrected word if a close match exists, None otherwise.
        """
        if not word:
            return None

        # Already a valid word — no correction needed
        if word in self.vocabulary:
            return None

        # Find the closest vocabulary word
        best_word = None
        min_dist = float("inf")

        for vocab_word in self.vocabulary:
            dist = damerau_levenshtein_distance(word, vocab_word)
            if dist < min_dist:
                min_dist = dist
                best_word = vocab_word

        # Only correct if the best match is close enough
        max_allowed = max(1.5, len(word) * 0.8)
        if min_dist <= max_allowed:
            return best_word

        return None
