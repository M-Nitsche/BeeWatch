import cv2 as cv
import numpy as np

cap = cv.VideoCapture("video_data/bees_demo.mp4")

ret, first_frame = cap.read()
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
mask = np.zeros_like(first_frame)
mask[..., 1] = 255 #image saturation
cv.namedWindow("Frame")


while(cap.isOpened()):
    ret, frame = cap.read()
    cv.imshow("Frame", frame)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Farneback Dense optical flow
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    mask[..., 0] = angle * 180 / np.pi / 2
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

    cv.imshow("Frame", rgb)
    prev_gray = gray
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()