"""
Place this script in the YoloV5 folder together with run_dection.py
"""


import argparse
import sys
import time
from pathlib import Path
import numpy as np
import imutils
import cv2
import torch
import torch.backends.cudnn as cudnn
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from run_detection import Detector
from centroid_tr import CentroidTracker

parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='./runs/train/exp7/weights/best.pt', help='model.pt path(s)')
parser.add_argument('--source', type=str, default='../dataset/video_data/bees_demo1.mp4', help='file/dir/URL/glob, 0 for webcam')
parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--view-img', default=False, action='store_true', help='show results')
parser.add_argument('--save-txt', default=True, action='store_true', help='save results to *.txt')
parser.add_argument('--save-conf', default=True, action='store_true', help='save confidences in --save-txt labels')
parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
parser.add_argument('--nosave', default=True, action='store_true', help='do not save images/videos')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--update', action='store_true', help='update all models')
parser.add_argument('--project', default='runs/detect', help='save results to project/name')
parser.add_argument('--name', default='exp', help='save results to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
opt = parser.parse_args()
### TRACKER
parser.add_argument('--maxDisappeared', default=5, help='maximum consecutive frames a given object is allowed to be marked as "disappeared"')
parser.add_argument('--save_tracking_img', default=True, help='if tracking image results and images should be safed')
parser.add_argument('--save_tracking_text', default=True, help='if tracking text results and images should be safed')
parser.add_argument('--show_tracking', default=True, help='view tracking')

args = parser.parse_args()
print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
check_requirements(exclude=('tensorboard', 'thop'))

# ToDO: place into funcitons

# create Detecor object
det = Detector(**vars(opt))

save_dir = det.save_dir
if args.save_tracking_img or args.save_tracking_text:
    # create folder tracking in yolo runs
    (save_dir / 'tracking').mkdir(parents=True, exist_ok=True)  # make dir

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker(args.maxDisappeared)
(H, W) = (None, None)

# Dataloader
if det.webcam:
    view_img = check_imshow()
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(opt.source, img_size=det.imgsz, stride=det.stride)
    bs = len(dataset)  # batch_size
else:
    dataset = LoadImages(opt.source, img_size=det.imgsz, stride=det.stride)
    bs = 1  # batch_size

# set mode
# print(dataset.__dict__["mode"])
# det.mode = dataset.__dict__["mode"]

counter = 0

for path, img, im0s, vid_cap in dataset:
    counter += 1
    # iterate over images
    res_list, img = det.run_detector_image(path, img, im0s, vid_cap, dataset)
    #print(res_list)

    frame = img # imutils.resize(img, width=1000)
    # if the frame dimensions are None, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    rects = []

    # loop over the detections
    for i in range(0, len(res_list)):
        #print(i)
        # filter out weak detections by ensuring the predicted
        # probability is greater than a minimum threshold
        if True:  # keine confidence
            # compute the (x, y)-coordinates of the bounding box for
            # the object, then update the bounding box rectangles list
            box_yolo = res_list[i][1:5] * np.array([W, H, W, H]) # center, size
            # for yolo-format
            box = np.array([0,0,0,0])
            box[0], box[1], box[2], box[3] = box_yolo[0] - (box_yolo[2]/2), box_yolo[1] - (box_yolo[3]/2), box_yolo[0] + (box_yolo[2]/2), box_yolo[1] + (box_yolo[3]/2)
            rects.append(box.astype("int"))
            # draw a bounding box surrounding the object so we can
            # visualize it
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)

    # update our centroid tracker using the computed set of bounding
    # box rectangles
    objects = ct.update(rects)
    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        if args.save_tracking_text:
            # save tracking CENTROID
            save_path = str(det.save_dir / 'tracking' / ("centroid_track_" + str(counter) + ".txt"))
            with open(save_path, 'a') as f:
                f.write((str(centroid[0]) + " " + str(centroid[1]) + " " + str(objectID) + '\n'))
    if args.show_tracking:
        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    if args.save_tracking_img:
        # tracking results will be saved in the yolo run folder
        save_path = str(det.save_dir / 'tracking' / ("centroid_track_img_" + str(counter) + ".png"))
        #img_path = save_path + dataset.count + ".png"
        cv2.imwrite(save_path, frame)

# do a bit of cleanup
cv2.destroyAllWindows()