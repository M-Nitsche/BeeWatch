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
if __name__!="__main__":
    from tracking.centroid_tr import CentroidTracker
else:
    from centroid_tr import CentroidTracker
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


def run_centroid_tracker(
        opt,
        # TRACKER
        args_tr
        ):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))

    # create Detecor object
    det = Detector(**vars(opt))

    # Define colors
    col_list = np.random.default_rng(42).random((100, 3)) * 255

    save_dir = det.save_dir
    if args_tr.save_tracking_img or args_tr.save_tracking_text:
        # create folder tracking in yolo runs
        (save_dir / 'tracking').mkdir(parents=True, exist_ok=True)  # make dir

    # initialize our centroid tracker and frame dimensions
    ct = CentroidTracker(args_tr.maxDisappeared)
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
    track_list = [[]]  # the placement in the list represents the ID
    sum_no_detected = 0  # sum over all detected bees
    list_id_tracked = []  # sum over all tracked bees

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

        # update our centroid tracker using the computed set of bounding
        # box rectangles
        objects = ct.update(rects)
        # loop over the tracked objects
        id_list_cur = []
        for (objectID, centroid) in objects.items():
            id_list_cur.append(objectID)
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            # place objectID in tracked id list
            if objectID not in list_id_tracked:
                list_id_tracked.append(objectID)
            # place centroid into track_list
            if len(track_list) > objectID:
                track_list[objectID].append([list(centroid)])
                #print(track_list[objectID], list(centroid), track_list[objectID].append(list(centroid)))
            else:
                while len(track_list) <= objectID:
                    track_list.append([])
                track_list[objectID].append([list(centroid)])
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col_list[(objectID%100)], 2) # id in centeriod
            cv2.circle(frame, (centroid[0], centroid[1]), 4, col_list[(objectID%100)], -1) # centeriod
            if args_tr.show_trajectories:
                #print("Center", centroid)
                #print(track_list[objectID][0][0])
                #print(np.array(track_list[objectID]))
                cv2.putText(frame, text, (track_list[objectID][0][0][0], track_list[objectID][0][0][1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col_list[(objectID%100)], 2) # ID at the start of the trajectory
                cv2.polylines(frame, np.array(track_list[objectID]),True,col_list[(objectID%100)], thickness=5) # trajectory
                #print("Here")
            if args_tr.save_tracking_text:
                # save tracking CENTROID
                save_path = str(det.save_dir / 'tracking' / ("centroid_track_" + str(counter) + ".txt"))
                with open(save_path, 'a') as f:
                    f.write((str(centroid[0]) + " " + str(centroid[1]) + " " + str(objectID) + '\n'))
        if args_tr.show_info:
            frame = cv2.copyMakeBorder(frame, 0, 40, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            text_det = "Detections " + str(len(res_list)) + " " # YOLO BB from object detection
            text_tr = "Tracking " + str(len(objects)) + " " # objects from Tracker
            text_sdet = "Total detections " + str(sum_no_detected) + " " # sum objects from detection
            text_str = "Total bees " + str(len(list_id_tracked)) + " " # sum no objects from Tracker
            l = len(text_det + text_tr + text_sdet + text_str)

            cv2.putText(frame, text_det, (int((W*0.01)), H + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0 ,0 ,0), 2)
            cv2.putText(frame, text_tr, (int((W*(len(text_det)/l))), H + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            cv2.putText(frame, text_sdet, (int((W*(len(text_det+text_tr)/l))), H + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            cv2.putText(frame, text_str, (int((W*(len(text_det+text_tr+text_sdet)/l))), H + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        if args_tr.show_tracking:
            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
        if args_tr.yield_back:
            # yield back also info: object number, current no. of bees by object det and tracker, sum up ...
            no_det_cur = len(res_list) # YOLO BB from object detection
            no_obj_cur = len(objects) # objects from Tracker
            # returns image, no of current objects tracking / object detection, lenght list of all ids tracked, sum of all detections, frame counter, list of current ids
            yield frame, no_det_cur, no_obj_cur, len(list_id_tracked), sum_no_detected, counter, id_list_cur
        if args_tr.save_tracking_img:
            # tracking results will be saved in the yolo run folder
            save_path = str(det.save_dir / 'tracking' / ("centroid_track_img_" + str(counter) + ".png"))
            #img_path = save_path + dataset.count + ".png"
            cv2.imwrite(save_path, frame)


    # do a bit of cleanup
    cv2.destroyAllWindows()

def arguments_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=parentdir_yolo+'/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=parentdir_yolo+'/data/bees_demo1.mp4', help='file/dir/URL/glob, 0 for webcam')
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

    ### TRACKER CENTRIOD
    parser.add_argument('--maxDisappeared', default=5,
                        help='maximum consecutive frames a given object is allowed to be marked as "disappeared"')
    parser.add_argument('--save_tracking_img', default=False,
                        help='if tracking image results and images should be saved')
    parser.add_argument('--save_tracking_text', default=False,
                        help='if tracking text results and images should be saved')
    parser.add_argument('--show_tracking', default=True, help='view tracking')
    parser.add_argument('--show_trajectories', default=True, help='view tracking trajectories')
    parser.add_argument('--show_info', default=True, help='yield back img and info for flask')
    parser.add_argument('--yield_back', default=False, help='yield back img and info for flask')
    args = parser.parse_args()

    return opt, args

if __name__=="__main__":
    # Parse arg
    opt, args = arguments_parse()
    # run_centroid_tracker is no always a generator
    for img in run_centroid_tracker(opt, args):
        print("IMAGE")




