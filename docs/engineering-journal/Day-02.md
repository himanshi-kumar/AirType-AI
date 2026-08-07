# Day 02

## Sprint

Sprint 1 — Webcam Engine

---

## Objective

Build the first Computer Vision application using OpenCV and understand how webcam frames are processed.

---

## Concepts Learned

- VideoCapture Object
- Webcam Pipeline
- Frame
- NumPy Array
- Image Shape
- Pixels
- BGR Color Space
- waitKey()
- release()
- destroyAllWindows()

---

## What I Built

- Connected laptop webcam
- Displayed live video feed
- Read frames continuously
- Successfully exited using keyboard input

---

## Bugs Faced

### Issue 1

Python command not found.

Reason:
macOS uses `python3`.

Solution:
Used `python3`.

---

### Issue 2

Wrong relative path while executing `main.py`.

Reason:
I executed `python src/main.py` while already inside the `src` directory.

Solution:
Learned how relative paths work.

---

### Issue 3

Confusion between files and folders.

Reason:
Used `cd main.py`.

Solution:
Understood that `cd` only changes directories.

---

## Key Learning

The biggest realization today was that a digital image is nothing but a NumPy array.

OpenCV does not understand objects; it only processes pixel values.

MediaPipe will later interpret those pixels into meaningful landmarks.

---

## Reflection

Today felt like building the foundation of a Computer Vision system. Instead of only learning syntax, I understood how webcam frames travel from the camera sensor to Python as NumPy arrays. This gives me confidence to move towards hand tracking using MediaPipe.