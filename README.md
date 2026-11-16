# ✨ Virtual Handwriting – Air Writing Using Hand Gestures

Write in the air without touching the screen! The webcam tracks your
index finger and uses it as a virtual pen.

## How it works
1. Webcam captures live video
2. Palm detection model finds the hand
3. Hand landmark model tracks 21 points
4. Index fingertip draws on a canvas

## Gestures
- Index finger up: write
- Index + middle finger up: pen lift

## Keys
r/g/b/y/m = color | +/- = brush size | s = save | c = clear | q = quit

## Tech Stack
Python, OpenCV, NumPy, ONNX Runtime, ONNX Models, VS Code

## How to run
pip install opencv-python numpy onnxruntime
python air.py

## Credits
ONNX models and helper files from OpenCV Zoo (MediaPipe models).




https://github.com/user-attachments/assets/bc23fbf8-568d-4fff-a139-c7f67c9dfa04



