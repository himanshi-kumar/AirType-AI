# AirType AI ✋⌨️

> An AI-powered touchless virtual keyboard — type using only hand gestures, no physical keyboard needed.

Built with **Python · OpenCV · MediaPipe · NumPy** by a 2nd-year CSE-AI student documenting every engineering decision from scratch.

---

## 🎯 What It Does

Point your index finger at a key → key highlights green.
Pinch index + thumb → key is typed.
Pinch on a word suggestion → full word auto-completes.

No touch. No hardware. Just a webcam and your hand.

---

## ✅ Features Built

| Feature | Sprint | Status |
|---------|--------|--------|
| Project setup, Git, venv | Sprint 0 | ✅ |
| OpenCV webcam engine | Sprint 1 | ✅ |
| Reusable OOP keyboard (`Key`, `Keyboard`) | Sprint 1 | ✅ |
| MediaPipe hand detection (21 landmarks) | Sprint 2 | ✅ |
| Finger tracking + landmark overlay | Sprint 2 | ✅ |
| Key hover detection (AABB collision) | Sprint 3 | ✅ |
| Pinch-to-click gesture (rising-edge + cooldown) | Sprint 3 | ✅ |
| Typed text display | Sprint 3 | ✅ |
| Word prediction (prefix completion + bigram) | Sprint 4 | ✅ |
| Suggestion bar (tap to auto-complete) | Sprint 4 | ✅ |
| Auto-correct | Sprint 5 | 🔄 |
| Voice output | Sprint 6 | ⬜ |
| Swipe typing | Sprint 7 | ⬜ |

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.12 | Core language |
| OpenCV 5.x | Webcam capture, frame processing, drawing |
| MediaPipe 0.10.x (Tasks API) | 21-point hand landmark detection |
| NumPy | Frame as array, pixel math |

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/himanshi-kumar/AirType-AI.git
cd AirType-AI

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download the MediaPipe model
curl -L -o assets/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

# Run
cd src
python3 main.py
```

**Press `Q` to quit.**

---

## 📂 Project Structure

```
AirType-AI/
│
├── assets/
│   └── hand_landmarker.task     ← MediaPipe ML model (7.5MB)
│
├── src/
│   ├── main.py                  ← Orchestrator (wires everything)
│   ├── hand_detector.py         ← MediaPipe Tasks API wrapper
│   ├── keyboard.py              ← Key, SuggestionBox, Keyboard classes
│   ├── gesture.py               ← PinchDetector (rising-edge + cooldown)
│   └── prediction.py            ← WordPredictor (prefix + bigram model)
│
├── docs/
│   ├── engineering-journal/     ← Day-01.md through Day-05.md
│   └── notes/                   ← Sprint summaries
│
├── CHANGELOG.md
├── requirements.txt
└── README.md
```

---

## 🧠 How It Works

### Hand Detection Pipeline
```
Webcam (BGR frame)
  → cv2.flip() [mirror]
  → BGR → RGB conversion
  → mp.Image wrapper
  → MediaPipe HandLandmarker.detect_for_video()
      ├── Stage 1: Palm Detector (finds bounding box)
      └── Stage 2: Landmark Model (21 keypoints)
  → 21 NormalizedLandmarks [0.0–1.0]
  → × frame_width/height → pixel coordinates
```

### Pinch-to-Click Mechanism
```
Euclidean distance(index_tip, thumb_tip) < 40px
  AND rising edge (first frame of pinch)
  AND cooldown == 0 (20 frames since last click)
  → CLICK fires
```

### Word Prediction
```
typed_text = "GOOD WOR"
  → current_partial = "WOR"
  → filter WORD_FREQUENCIES for words starting with "WOR"
  → sort by frequency
  → ["WORLD", "WORK", "WORD"]

typed_text = "GOOD "
  → last_complete = "GOOD"
  → BIGRAMS["GOOD"] = ["MORNING", "NIGHT", "LUCK"]
  → suggestions = ["MORNING", "NIGHT", "LUCK"]
```

---

## 📖 Engineering Journal

Every concept, bug, and decision documented day by day:

| Day | Topic |
|-----|-------|
| [Day 01](docs/engineering-journal/Day-01.md) | Project setup, OpenCV basics, Git init |
| [Day 02](docs/engineering-journal/Day-02.md) | Webcam engine, frames as NumPy arrays |
| [Day 03](docs/engineering-journal/Day-03.md) | MediaPipe API migration (`mp.solutions` → Tasks API) |
| [Day 04](docs/engineering-journal/Day-04.md) | AABB collision, Euclidean pinch, rising-edge detection |
| [Day 05](docs/engineering-journal/Day-05.md) | N-gram prediction, bigram model, suggestion bar |

---

## 📈 Commit History

```
feat: word prediction engine with bigram model and suggestion bar
feat: finger tracking, hover detection, and pinch-to-click typing
feat: integrate MediaPipe Tasks API for hand detection
feat: implement reusable keyboard UI using OOP
docs: add Sprint 1 documentation and engineering journal
feat: initialize AirType AI project structure
```

---

## 🔑 Key Algorithms

| Algorithm | Used For | File |
|-----------|----------|------|
| AABB collision | Is finger over a key? | `keyboard.py` |
| Euclidean distance | Is pinch happening? | `gesture.py` |
| Rising-edge detection | Fire click once per pinch | `gesture.py` |
| Prefix frequency matching | Word completion | `prediction.py` |
| Bigram Markov model | Next-word prediction | `prediction.py` |
| MediaPipe Tasks API | 21-point hand landmarks | `hand_detector.py` |

---

*Built sprint by sprint. Every line documented. Every decision explained.*