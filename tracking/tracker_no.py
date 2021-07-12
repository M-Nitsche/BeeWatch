"""
Place this script in the tracking folder together with run_dection.py
"""

import os, sys
import argparse
import sys
# import time
from pathlib import Path
import numpy as np
# import imutils
import cv2
import torch
import torch.backends.cudnn as cudnn
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
print("Parent dir", parentdir)
parentdir_yolo = parentdir + '/yolov5/'
sys.path.append(parentdir_yolo)
# from models.experimental import attempt_load
from utils.datasets import LoadWebcam_Jetson, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
# from utils.plots import colors, plot_one_box
# from utils.torch_utils import select_device, load_classifier, time_synchronized

from run_detection import Detector


def run_no_tracker(
        opt,
        args_tr
        ):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))

    # create Detecor object
    det = Detector(**vars(opt))

    (H, W) = (None, None)
    # Dataloader
    if opt.source == "Camera":
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadWebcam_Jetson(img_size=det.imgsz, stride=det.stride)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(opt.source, img_size=det.imgsz, stride=det.stride)
        bs = 1  # batch_size   

    # set mode
    # print(dataset.__dict__["mode"])
    # det.mode = dataset.__dict__["mode"]

    counter = 0
    sum_no_detected = 0  # sum over all detected bees
    #print("###################################################", dataset)
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
            sum_no_detected += 1
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
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 255), 2)
        if args_tr.show_info:
            frame = cv2.copyMakeBorder(frame, 0, 40, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            text_det = "Detections " + str(len(res_list)) + " " # YOLO BB from object detection
            text_sdet = "Total detections " + str(sum_no_detected) + " " # sum objects from detection
            l = len(text_det + text_sdet)

            cv2.putText(frame, text_det, (int((W*0.01)), H + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0 ,0 ,0), 2)
            cv2.putText(frame, text_sdet, (int((W*(len(text_det)/l))), H + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        if args_tr.show_tracking:
            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
        if args_tr.yield_back:
            # yield back image, no of current detections, sum of detections and frame no
            yield frame, len(res_list), sum_no_detected, counter

    # do a bit of cleanup
    cv2.destroyAllWindows()

def arguments_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=parentdir_yolo+'/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='Camera', help='file/dir/URL/glob, 0 for webcam') #parentdir_yolo+'/data/bees_demo1.mp4'
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default=False, action='store_true', help='show results')
    parser.add_argument('--save-txt', default=False, action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', default=False, action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', default=True, action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=parentdir_yolo+'/runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()

    ### TRACKER CENTERIOD
    parser.add_argument('--show_tracking', default=True, help='view tracking')
    parser.add_argument('--show_info', default=True, help='yield back img and info for flask')
    parser.add_argument('--yield_back', default=False, help='yield back img and info for flask')
    args = parser.parse_args()

    return opt, args

if __name__=="__main__":
    # Parse arg
    opt, args = arguments_parse()
    # run_centroid_tracker is no always a generator
    for img in run_no_tracker(opt, args):
        print("IMAGE")




