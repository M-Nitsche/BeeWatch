import cv2
import os
VIDEO_NAME = "bee_video"
VIDEO_DIR = "video_data"
vidcap = cv2.VideoCapture(os.path.join(VIDEO_DIR, VIDEO_NAME+ ".mp4"))
success,image = vidcap.read()
count = 0

SPLIT_OUTPUT_DIR = os.path.join(VIDEO_DIR, VIDEO_NAME + "_split_bg")

fgbg = cv2.createBackgroundSubtractorMOG2()

if not os.path.exists(SPLIT_OUTPUT_DIR):
    os.mkdir(SPLIT_OUTPUT_DIR)

while success:
    fgmask = fgbg.apply(image)
    cv2.imwrite(os.path.join(SPLIT_OUTPUT_DIR, "%s_frame%d.jpg" % (VIDEO_NAME, count)), fgmask)     # save frame as JPEG file
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1