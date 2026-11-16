import os
import cv2 as cv
import numpy as np
from mp_palmdet import MPPalmDet
from mp_handpose import MPHandPose

here = os.path.dirname(os.path.abspath(__file__))
be, tg = cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU

palm = MPPalmDet(
    modelPath=os.path.join(here, "palm_detection_mediapipe_2023feb.onnx"),
    nmsThreshold=0.3, scoreThreshold=0.6, topK=1,
    backendId=be, targetId=tg)
hand = MPHandPose(
    modelPath=os.path.join(here, "handpose_estimation_mediapipe_2023feb.onnx"),
    confThreshold=0.9, backendId=be, targetId=tg)

colors = {
    ord('r'): (0, 0, 255),
    ord('g'): (0, 255, 0),
    ord('b'): (255, 0, 0),
    ord('y'): (0, 255, 255),
    ord('m'): (255, 0, 255),
}
color = (255, 0, 255)
thickness = 5

cap = cv.VideoCapture(0)
canvas = None
prev = None
count = 1

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv.flip(frame, 1)
    if canvas is None:
        canvas = np.zeros_like(frame)

    drawing = False
    for p in palm.infer(frame):
        res = hand.infer(frame, p)
        if res is None:
            continue
        lm = res[4:67].reshape(21, 3)
        x, y = int(lm[8][0]), int(lm[8][1])
        index_up = lm[8][1] < lm[6][1]
        middle_up = lm[12][1] < lm[10][1]
        if index_up and not middle_up:
            drawing = True
            if prev is not None:
                cv.line(canvas, prev, (x, y), color, thickness)
            prev = (x, y)
        cv.circle(frame, (x, y), 8, (0, 255, 0), -1)
    if not drawing:
        prev = None

    # status bar
    cv.rectangle(frame, (0, 0), (frame.shape[1], 30), (0, 0, 0), -1)
    cv.putText(frame, f"Size:{thickness}  r/g/b/y/m=color  +/-=size  s=save  c=clear  q=quit",
               (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv.circle(frame, (frame.shape[1] - 20, 15), 8, color, -1)

    cv.imshow("Air Writing", cv.add(frame, canvas))
    k = cv.waitKey(1) & 0xFF

    if k in colors:
        color = colors[k]
    elif k in (ord('+'), ord('=')):
        thickness = min(thickness + 1, 30)
    elif k == ord('-'):
        thickness = max(thickness - 1, 1)
    elif k == ord('c'):
        canvas[:] = 0
    elif k == ord('s'):
        out = np.full_like(canvas, 255)          # white background
        mask = canvas.any(axis=2)
        out[mask] = canvas[mask]
        path = os.path.join(here, f"air_wriqting_{count}.png")
        cv.imwrite(path, out)
        print("Saved:", path)
        count += 1
    elif k == ord('q'):
        break

cap.release()
cv.destroyAllWindows()