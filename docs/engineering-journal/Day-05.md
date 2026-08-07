# Day 05

## Sprint

Sprint 4 — Word Prediction Engine + Suggestion Bar

---

## Objective

Make AirType AI smart: suggest word completions and next-word predictions as the user types.

---

## What I Built

### New Files

| File | Purpose |
|------|---------|
| `src/prediction.py` | `WordPredictor` — prefix completion + bigram next-word prediction |

### Modified Files

| File | Changes |
|------|---------|
| `src/keyboard.py` | `SuggestionBox` class, suggestion bar, `set_suggestions()`, `select_suggestion()`, 1280×720 layout |
| `src/main.py` | 1280×720 resolution, `WordPredictor` wired into pipeline, suggestion priority in pinch handler |

---

## Concepts Learned

### 1. Prefix-Based Word Completion

The predictor stores a dictionary:
```python
WORD_FREQUENCIES = {"HAVE": 88, "HAPPY": 18, "HAD": 45, ...}
```

Given prefix "HA":
1. Filter: keep only words starting with "HA"
2. Sort by frequency (descending)
3. Return top 3: ["HAVE", "HAD", "HAPPY"]

Time complexity: O(V log V) where V = vocabulary size (~300 words).
For V=300, this is ~2,400 operations — runs in microseconds.

Upgrade path: Replace the filter+sort with a **Trie** (prefix tree) for O(L) lookup,
where L = prefix length. Used in production autocomplete systems (Google Search, VS Code IntelliSense).

### 2. N-gram Language Model (Bigram)

A bigram model predicts the next word given the current word:

```
P(next | current) = count(current → next) / count(current)
```

Example bigrams in our model:
```python
BIGRAMS["GOOD"] = ["MORNING", "NIGHT", "LUCK", "JOB"]
BIGRAMS["I"]    = ["AM", "HAVE", "WILL", "THINK"]
```

**This is the foundation of every language model:**
- Bigram: P(w₂ | w₁)
- Trigram: P(w₃ | w₁, w₂)
- GPT-4: P(wₙ | w₁...wₙ₋₁) — considers the entire history with billions of parameters

Our bigram model with 300 words is the simplest version of what GPT does.

### 3. NLP Backoff (Graceful Degradation)

If the user types a word not in our bigram table (e.g., "ZEBRA"):
```python
if last_word in self.bigrams:
    return self.bigrams[last_word][:n]
# Backoff:
return self._top_n_words(n)  # ["THE", "BE", "TO"]
```

This is called **backoff** — if a specific model fails, fall back to a simpler one.

Production systems use **Kneser-Ney smoothing**, which mathematically blends
higher-order and lower-order n-grams weighted by their reliability.

### 4. Word Auto-Complete (String Manipulation)

When the user pinches on "WORLD" while having typed "HEL":

```python
typed_text = "I LOVE HEL"
words = typed_text.split(" ")   # ["I", "LOVE", "HEL"]
words = words[:-1]              # ["I", "LOVE"]  ← drop partial word
words.append("WORLD")          # ["I", "LOVE", "WORLD"]
typed_text = " ".join(words) + " "  # "I LOVE WORLD "
```

The trailing space signals "word is done, start next-word prediction".

### 5. Priority in Event Handling

When a pinch fires, we check in order:
```python
if keyboard.hovered_suggestion:
    keyboard.select_suggestion()   # priority 1: auto-complete
elif keyboard.hovered_key:
    keyboard.register_click()      # priority 2: type letter
```

This is called **event priority** — UI systems always have a defined order
for handling overlapping interactive regions. In web dev this is called
**z-index** and **event bubbling**.

### 6. Resolution Normalization

```python
if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
```

Why? MediaPipe returns landmark coordinates as fractions [0,1].
We multiply by frame width/height to get pixels.
If the frame is 480px wide but our keyboard expects 1280px, everything is misaligned.
Normalizing to a fixed size guarantees all coordinate math is consistent.

---

## Test Results

```
Prefix HA:           ['HAVE', 'HAPPEN', 'HAND']
Prefix WOR:          ['WORLD', 'WORK', 'WORD']
After GOOD:          ['MORNING', 'NIGHT', 'LUCK']
Mid-word HELLO WOR:  ['WORLD', 'WORK', 'WORD']
Empty input:         ['THE', 'BE', 'TO']
All Sprint 4 tests:  PASS
```

---

## Full Pipeline (Sprint 4)

```
Frame
  ↓ flip
  ↓ MediaPipe inference → landmarks
  ↓ get_landmark_position(INDEX_TIP, THUMB_TIP)
  ↓ predictor.get_suggestions(typed_text)  ← NEW
  ↓ keyboard.set_suggestions(words)        ← NEW
  ↓ keyboard.draw(frame, finger_pos)
      ├── draw suggestion bar (3 blue boxes)  ← NEW
      ├── draw text display bar
      └── draw 28 keys (green if hovered)
  ↓ pinch.update() → clicked?
      ├── hovered_suggestion? → select_suggestion()  ← NEW (priority 1)
      └── hovered_key?        → register_click()     (priority 2)
  ↓ imshow
```

---

## Reflection

Today AirType became intelligent. The keyboard no longer just types letters — it
anticipates what you're going to say.

The most important insight: the same algorithm powering Google Search autocomplete
(prefix matching with frequency ranking) can be implemented in 20 lines of Python.
GPT-4 is the same idea, scaled up by a factor of 10 billion parameters.

Understanding the simple version first makes the complex version understandable.
