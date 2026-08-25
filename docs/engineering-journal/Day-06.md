# Day 06

## Sprint

Sprint 5 — Spelling Auto-Correct (Damerau-Levenshtein + QWERTY Weights)

---

## Objective

Make AirType AI forgiving: automatically correct misspelled words when the user presses Space, and show fuzzy suggestions even when the typed prefix doesn't exactly match any word.

---

## What I Built

### Modified Files

| File | Changes |
|------|---------|
| `src/prediction.py` | `KEY_CENTERS` dict, `get_substitution_cost()`, `damerau_levenshtein_distance()`, `get_autocorrect()` method, fuzzy backoff in `_predict_completions()` |
| `src/keyboard.py` | `register_click(predictor)` — auto-corrects last word on SPC press |
| `src/main.py` | Passes `predictor` into `keyboard.register_click(predictor)` |

### New Files

| File | Purpose |
|------|---------|
| `tests/test_spelling.py` | 30 unit tests covering substitution cost, DL distance, autocorrect, fuzzy suggestions, and keyboard integration |

---

## Concepts Learned

### 1. Levenshtein Distance (Edit Distance)

The minimum number of single-character edits to transform one string into another:
- **Insertion**: "HELO" → "HE**L**LO" (insert L)
- **Deletion**: "HELLOO" → "HELLO" (delete O)
- **Substitution**: "TH**W**" → "TH**E**" (substitute W→E)

```
dp[i][j] = min(
    dp[i-1][j]   + 1,           # deletion
    dp[i][j-1]   + 1,           # insertion
    dp[i-1][j-1] + sub_cost     # substitution
)
```

Time complexity: O(m × n) where m, n are string lengths.

### 2. Damerau-Levenshtein (Adding Transpositions)

Damerau observed that **80% of all human typing errors** are one of four types:
1. Insertion of a character
2. Deletion of a character
3. Substitution of a character
4. **Transposition** of two adjacent characters ← NEW

Standard Levenshtein doesn't handle transpositions natively. "HPAP" → "HAPPY" costs 3 edits without transpositions but only 2 with them (swap P↔A, insert Y).

```python
# Transposition check: if chars form a swap pattern
if s1[i-1] == s2[j-2] and s1[i-2] == s2[j-1]:
    dp[i][j] = min(dp[i][j], dp[i-2][j-2] + 1.0)
```

### 3. QWERTY-Weighted Substitution Cost

Standard edit distance treats all substitutions equally (cost = 1). But on a physical keyboard, adjacent key errors are more likely than distant ones.

We compute Euclidean distance between key centers:
```python
pixel_dist = sqrt((x1 - x2)² + (y1 - y2)²)
cost = 0.2 + 0.8 × min(2.0, pixel_dist / 90.0)
```

Results:
- W→E (adjacent): cost ≈ 0.79
- Q→S (diagonal): cost ≈ 1.18
- Q→M (far apart): cost = 1.80

This means the corrector understands **which typos are physically plausible** based on our QWERTY layout.

### 4. First-Letter Penalty

Users rarely mistype the first letter of a word. Adding a +1.5 penalty for first-letter mismatches prevents short unrelated words from stealing suggestions:

Without penalty: "COMPUTW" → "OUT" (distance 3.7) beats "COMPUTER" (distance 4.0)
With penalty: "COMPUTW" → "COMPUTER" (distance 0.79) wins easily

### 5. Fuzzy Backoff in Suggestions

Sprint 4's `_predict_completions()` only returned exact prefix matches:
```
Input: "THW" → matches: [] (nothing starts with "THW")
```

Sprint 5 adds a **fuzzy backoff**: if prefix matching yields fewer than N results, fill the remaining slots with Damerau-Levenshtein matches:
```
Input: "THW" → prefix: [] → fuzzy: ["THE", "THEY", "THEN"]
```

This is the same principle as NLP backoff from Sprint 4:
- Sprint 4: bigram → backoff to top frequency
- Sprint 5: prefix → backoff to edit distance

### 6. Auto-Correct on Space

When the user presses Space, the keyboard extracts the last word and queries `predictor.get_autocorrect()`. If the word is misspelled and a close match exists, it's silently replaced:

```
User types: "I AM THW" + SPC
Pipeline:   last_word = "THW"
            correction = predictor.get_autocorrect("THW") → "THE"
            typed_text = "I AM THE "
```

The trailing space then triggers next-word prediction as in Sprint 4.

---

## Test Results

```
.venv/bin/python -m unittest tests.test_spelling -v

Ran 30 tests in 0.019s
OK

Key results:
  THW → THE (adjacent key correction)
  HPAP → HAPPY (transposition + insertion)
  COMPUTW → COMPUTER (adjacent key correction)
  HAPY → HAPPY (missing letter)
  HELLO → HELLO (valid word, no change)
  XZQWJ → None (gibberish rejected)
```

---

## Full Pipeline (Sprint 5)

```
Frame
  ↓ flip
  ↓ MediaPipe inference → landmarks
  ↓ get_landmark_position(INDEX_TIP, THUMB_TIP)
  ↓ predictor.get_suggestions(typed_text)
  ↓   ├── prefix matches? → return them
  ↓   └── NOT ENOUGH? → fuzzy DL backoff ← NEW
  ↓ keyboard.set_suggestions(words)
  ↓ keyboard.draw(frame, finger_pos)
  ↓ pinch.update() → clicked?
      ├── hovered_suggestion? → select_suggestion()
      └── hovered_key?
            ├── SPC? → auto-correct last word + append space ← NEW
            ├── BACK? → delete last char
            └── other → append letter
  ↓ imshow
```

---

## Reflection

Today AirType learned to forgive mistakes. The distance algorithm understands the physical layout of our keyboard — adjacent-key typos are treated as cheap errors, while random character substitutions are heavily penalized. Transposition support handles the most common human typing error (swapping adjacent letters).

The most important architectural insight: by keeping `get_autocorrect()` as a method on `WordPredictor` (not inside `Keyboard`), we maintained the separation of concerns established in Sprint 4. The keyboard knows *when* to ask for corrections (on Space press), and the predictor knows *how* to compute them. Neither depends on the other's internals.
