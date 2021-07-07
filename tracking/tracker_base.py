"""
Place this script in the YoloV5 folder together with run_dection.py
"""


import argparse
import sys
import time
from pathlib import Path

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

parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='C:/Users/ojkbe/PycharmProjects/OpticalFlowTracking/yolov5/best.pt', help='model.pt path(s)')
parser.add_argument('--source', type=str, default='C:/Users/ojkbe/PycharmProjects/OpticalFlowTracking/yolov5/bees_demo1.mp4', help='file/dir/URL/glob, 0 for webcam')
parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--view-img', action='store_true', help='show results')
parser.add_argument('--save-txt', default=True, action='store_true', help='save results to *.txt')
parser.add_argument('--save-conf', default=True, action='store_true', help='save confidences in --save-txt labels')
parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
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

print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
check_requirements(exclude=('tensorboard', 'thop'))

# ToDO: place into funcitons

# create Detecor object
det = Detector(**vars(opt))

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

for path, img, im0s, vid_cap in dataset:
    # iterate over images
    res_list, img = det.run_detector_image(path, img, im0s, vid_cap, dataset)
    print(res_list)