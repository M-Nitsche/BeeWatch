import numpy as np
import cv2 as cv

#cap = cv2.VideoCapture(0)
cap = cv.VideoCapture("video_data/bees_demo.mp4")

cv.namedWindow("Frame")

while True:
    _, frame = cap.read()
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    corners = cv.goodFeaturesToTrack(gray_frame, 25, 0.01, 10)
    corners = np.int0(corners)

    for i in corners:
        x, y = i.ravel()
        cv.circle(frame, (x, y), 3, 255, -1)

    cv.imshow("Frame", frame)
    key_input = cv.waitKey(1)
    if key_input == 27:
        break

cap.release()
cv.destroyAllWindows()
