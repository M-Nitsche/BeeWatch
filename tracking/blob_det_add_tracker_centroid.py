"""
Place this script in the tracking folder
"""
import os, sys
import argparse
import sys
# import time
from pathlib import Path
from datetime import datetime
import numpy as np
# import imutils
import cv2
import torch
import torch.backends.cudnn as cudnn
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
if __name__!="__main__":
    from tracking.centroid import CentroidTracker
else:
    from centroid import CentroidTracker
print("Parent dir", parentdir)
parentdir_yolo = parentdir + '/yolov5/'
sys.path.append(parentdir_yolo)
# from models.experimental import attempt_load
from utils.datasets import LoadWebcam_Jetson, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
# from utils.plots import colors, plot_one_box
# from utils.torch_utils import select_device, load_classifier, time_synchronized
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance as dist
from run_detection import Detector

def init_blob_det(args_tr):
    # Set up the parameters for the detector.
    params = cv2.SimpleBlobDetector_Params()

    params.blobColor = 255
    params.filterByColor = True

    params.minArea = 300
    params.maxArea = 10000
    params.filterByArea = True

    params.minThreshold = 1
    params.maxThreshold = 255

    params.filterByInertia = False
    params.filterByConvexity = False

    params.filterByCircularity = True
    params.minCircularity = 0.1

    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    return detector


def run_blob_det_add_centroid_tracker(
        opt,
        # TRACKER
        args_tr
        ):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))

    # Object for BackgroundSubstraction
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

    # Blob detector object
    blob = init_blob_det(args_tr)

    if args_tr.det_and_blob:
        # create Detecor object
        det = Detector(**vars(opt))
        save_dir = det.save_dir
    else:
        save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run

    # Define colors
    col_list = np.random.default_rng(42).random((100, 3)) * 255

    if args_tr.save_tracking_img or args_tr.save_tracking_text:
        # create folder tracking in yolo runs
        (save_dir / 'tracking').mkdir(parents=True, exist_ok=True)  # make dir

    # initialize our centroid tracker and frame dimensions
    ct = CentroidTracker(args_tr.maxDisappeared, args_tr.tracker_threshold)
    (H, W) = (None, None)

    # Dataloader
    if opt.source == "Camera":
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        if not args_tr.det_and_blob:
            dataset = LoadWebcam_Jetson(img_size=opt.imgsz)
        else:
            dataset = LoadWebcam_Jetson(img_size=det.imgsz, stride=det.stride)
        bs = len(dataset)  # batch_size
    else:
        if not args_tr.det_and_blob:
            dataset = LoadImages(opt.source, img_size=opt.imgsz)
        else:
            dataset = LoadImages(opt.source, img_size=det.imgsz, stride=det.stride)
        bs = 1  # batch_size

    # set mode
    # print(dataset.__dict__["mode"])
    # det.mode = dataset.__dict__["mode"]

    counter = 0
    det_results_calc = False
    track_list = [[]]  # the placement in the list represents the ID
    sum_no_detected = 0  # sum over all detected bees
    list_id_tracked = []  # sum over all tracked bees
    blacklist_id = []  # blacklist of ids
    for path, img, im0s, vid_cap in dataset:
        counter += 1
        # if the frame dimensions are None, grab them
        if W is None or H is None:
            (H, W) = im0s.shape[:2]
        # iterate over images
        # blob detection time
        start_time = datetime.now()
        ################################# Blob detection ##################################
        bs_img = fgbg.apply(im0s)
        keypoints = blob.detect(bs_img)
        blob_res_list, cen_blob_res_list = [], []
        for k in keypoints:
            # convert keypoints to yolo format
            c_x, c_y, w, h = k.pt[0], k.pt[1], k.size, k.size
            #print("Centriods of blob detection ", c_x, c_y)
            blob_res_list.append([c_x - w/2, c_y - h/2, c_x + w/2, c_y + h/2])
            cen_blob_res_list.append([c_x, c_y])
        ###################################################################################
        diff_time = (datetime.now() - start_time).seconds + ((datetime.now() - start_time).microseconds) / 1000000
        print("Blob Detection took: ", diff_time, " seconds")

        # tracking time
        start_time = datetime.now()
        ###################### Check if Obj det should be performed #######################
        # obj det is performed before disappearance takes place and if nothing goes back
        # from Blob detection - both are these are checked below
        # here check if new IDs would be registered with the BB from blob det
        # these must then be tested if these are correct
        new_ids_to_register = False
        if args_tr.det_and_blob:
            new_ids_to_register = ct.new_id_registered(blob_res_list)
        ###################################################################################

        ################################ Object detection #################################
        det_res_list, cen_det, update_list = [], [], []
        diff_time_obj = 0
        if args_tr.det_and_blob:
            if new_ids_to_register or blob_res_list == [] or counter % args_tr.maxDisappeared == 0:
                print(counter, "object detection run")
                print("New id", new_ids_to_register)
                start_time_obj = datetime.now()
                res_list, img = det.run_detector_image(path, img, im0s, vid_cap, dataset)
                diff_time_obj = (datetime.now() - start_time_obj).seconds + (
                    (datetime.now() - start_time_obj).microseconds) / 1000000
                print("Object Detection, saving, drawing, showing took: ", diff_time_obj, " seconds")

                for i, ybb in enumerate(res_list):
                    sum_no_detected += 1
                    # compute the (x, y)-coordinates of the bounding box for
                    # the object, then update the bounding box rectangles list
                    box_yolo = ybb[1:5] * np.array([W, H, W, H])  # center, size
                    cen_det.append([box_yolo[0], box_yolo[1]])
                    # for yolo-format
                    box = np.array([0,0,0,0])
                    box[0], box[1], box[2], box[3] = box_yolo[0] - (box_yolo[2] / 2), box_yolo[1] - (
                                    box_yolo[3] / 2), \
                                                         box_yolo[0] + (box_yolo[2] / 2), box_yolo[1] + (
                                                                     box_yolo[3] / 2)
                    det_res_list.append(box.tolist())
                    box.astype("int")
                    (startX, startY, endX, endY) = box[0], box[1], box[2], box[3]
                    # draw BB on image
                    cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 255), 2)

                cen_blob_res_list, cen_det = np.array(cen_blob_res_list), np.array(cen_det)
                if cen_blob_res_list.size == 0 or cen_det.size == 0:
                    print("No Det or no Blobs")
                    # one or both are empty and have no detections
                    # nothing to compare
                    # if blob detection / object tracking empty then update with object detections
                    if cen_blob_res_list.size == 0 and cen_det.size != 0:
                        print("Update tracker after object detection - no Blob but obj detection")
                        update_list = det_res_list
                    # if object detection empty nothing to update already updated
                else:
                    print("Calculate distance. No of Blobs", len(cen_blob_res_list), "No of obj det", len(cen_det))
                    D = dist.cdist(cen_blob_res_list, cen_det)  # D is structured as blob[det]
                    cand_ids, _filter_ids = linear_sum_assignment(D,
                                                                  maximize=False)  # cand_dis is blob for det in _filter_ids
                    cen_blob_res_list = cen_blob_res_list.tolist()
                    # if blob detections do not match do not loose those
                    index_to_delete = []
                    index_save_blob_anno = []

                    # dont match everything together - matches higher than a threshold are invalid
                    # discontinue blob det / id and add det to tracker
                    for i in range(0, len(cand_ids)):
                        if D[cand_ids[i], _filter_ids[i]] > args_tr.matching_threshold:
                            # blacklist_id.append(id_list[i])
                            print("delete blob too far away", cand_ids[i], "distance",
                                  D[cand_ids[i], _filter_ids[i]],
                                  "threshold", args_tr.matching_threshold)
                            # add not matched obj det to the list
                            update_list.append(det_res_list[_filter_ids[i]])
                            # do not delete not matched blob
                        else:
                            # already det matched to blob
                            # matched obj det is better than blob detection, swap these out
                            index_to_delete.append(cand_ids[i])
                            update_list.append(det_res_list[_filter_ids[i]])
                            index_save_blob_anno.append(cand_ids[i])
                    updated_blob_list = []
                    updated_blob_cen_list = []
                    for idx, b in enumerate(blob_res_list):
                        if idx not in index_to_delete:
                            updated_blob_list.append(b)
                            updated_blob_cen_list.append(cen_blob_res_list[idx])
                        elif idx in index_save_blob_anno:
                            updated_blob_cen_list.append(cen_blob_res_list[idx])
                    blob_res_list = updated_blob_list
                    cen_blob_res_list = updated_blob_cen_list

                    # if object detection finds new bees add those
                    # if in _filter_ids det is missing -> not represented in blob/id detection add to tracking
                    for i in range(0, len(cen_det)):
                        if i not in _filter_ids:
                            print("Found BB from obj det that is not represented in Blob detection")
                            update_list.append(det_res_list[i])

                frame = img
                ###################################################################################
            else:
                print(counter, "no object detection run")
                frame = im0s
        else:
            frame = im0s
        ###################################################################################

        ################################ Update Tracking #################################
        for b in blob_res_list:
            update_list.append(b)
        objects = ct.update(update_list)
        # get current IDs
        cur_id_list = list(ct.objects.keys())
        ###################################################################################

        if args_tr.show_blob_update:
            for cen in cen_blob_res_list:
                # draw blob update
                cv2.circle(frame, (int(cen[0]), int(cen[1])), 6, (0, 255, 0), 2)

        # loop over the tracked objects
        id_list_cur = []
        for (objectID, centroid) in objects.items():
            if objectID not in list_id_tracked:
                list_id_tracked.append(objectID)
            id_list_cur.append(objectID)
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            # place centroid into track_list
            if len(track_list) > objectID:
                track_list[objectID].append([list(centroid)])
                # print(track_list[objectID], list(centroid), track_list[objectID].append(list(centroid)))
            else:
                while len(track_list) <= objectID:
                    track_list.append([])
                track_list[objectID].append([list(centroid)])
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col_list[(objectID % 100)], 2)  # id in centeriod
            cv2.circle(frame, (centroid[0], centroid[1]), 4, col_list[(objectID % 100)], -1)  # centeriod
            if args_tr.show_trajectories:
                # print("Center", centroid)
                # print(track_list[objectID][0][0])
                # print(np.array(track_list[objectID]))
                cv2.putText(frame, text, (track_list[objectID][0][0][0], track_list[objectID][0][0][1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col_list[(objectID % 100)],
                            2)  # ID at the start of the trajectory
                cv2.polylines(frame, np.array(track_list[objectID]), True, col_list[(objectID % 100)],
                              thickness=5)  # trajectory
                # print("Here")
            if args_tr.save_tracking_text:
                # save tracking CENTROID
                save_path = str(save_dir / 'tracking' / ("centroid_track_" + str(counter) + ".txt"))
                with open(save_path, 'a') as f:
                    f.write((str(centroid[0]) + " " + str(centroid[1]) + " " + str(objectID) + '\n'))
        if args_tr.show_info:
            frame = cv2.copyMakeBorder(frame, 0, 40, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            text_det = "Detections " + str(len(cen_det)) + " "  # YOLO BB from object detection
            text_tr = "Tracking " + str(len(id_list_cur)) + " "  # objects from Tracker
            text_sdet = "Total detections " + str(sum_no_detected) + " "  # sum objects from detection
            text_str = "Total bees " + str(len(list_id_tracked)) + " "  # sum no objects from Tracker
            text_blob = "Blobs " + str(len(cen_blob_res_list)) + " "  # no of blobs detected
            l = len(text_det + text_tr + text_sdet + text_str + text_blob)

            cv2.putText(frame, text_det, (int((W * 0.01)), H + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            cv2.putText(frame, text_tr, (int((W * (len(text_det) / l))), H + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 0, 0), 2)
            cv2.putText(frame, text_blob, (int((W * (len(text_det + text_tr) / l))), H + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 0, 0), 2)
            cv2.putText(frame, text_sdet, (int((W * (len(text_det + text_blob + text_tr) / l))), H + 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 0), 2)
            cv2.putText(frame, text_str, (int((W * (len(text_det + text_blob + text_tr + text_sdet) / l))), H + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        if args_tr.show_tracking:
            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
        if args_tr.yield_back:
            # yield back also info: object number, current no. of bees by object det and tracker, sum up ...
            no_det_cur = len(cen_det)  # YOLO BB from object detection
            no_obj_cur = len(id_list_cur)  # objects from Tracker
            # returns image, no of current objects tracking / object detection, lenght list of all ids tracked, sum of all detections, frame counter, list of current ids
            yield frame, no_det_cur, no_obj_cur, len(list_id_tracked), sum_no_detected, counter, id_list_cur
        if args_tr.save_tracking_img:
            # tracking results will be saved in the yolo run folder
            save_path = str(save_dir / 'tracking' / ("centroid_track_img_" + str(counter) + ".png"))
            # img_path = save_path + dataset.count + ".png"
            cv2.imwrite(save_path, frame)
        diff_time = ((datetime.now() - start_time).seconds + ((datetime.now() - start_time).microseconds) / 1000000) - diff_time_obj
        print("Tracking, saving, drawing, showing took: ", diff_time, " seconds")

        # do a bit of cleanup
    cv2.destroyAllWindows()

def arguments_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=parentdir_yolo + '/best.pt',
                        help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=parentdir_yolo + '/data/bees_demo1.mp4',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default=False, action='store_true', help='show results')
    parser.add_argument('--save-txt', default=False, action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', default=False, action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', default=True, action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=parentdir_yolo + '/runs/detect', help='save results to project/name')
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
    parser.add_argument('--save_tracking_img', default=True,
                        help='if tracking image results and images should be saved')
    parser.add_argument('--save_tracking_text', default=False,
                        help='if tracking text results and images should be saved')
    parser.add_argument('--show_tracking', default=True, help='view tracking')
    parser.add_argument('--show_trajectories', default=True, help='view tracking trajectories')
    parser.add_argument('--show_info', default=True, help='show information on the bottom of the image')
    parser.add_argument('--yield_back', default=False, help='yield back img and info for flask')
    parser.add_argument('--det_and_blob', default=True,
                        help='when using blob detection add object det as a checker')
    parser.add_argument('--matching_threshold', default=100,
                        help='threshold between detections from blob detection and object detection')
    parser.add_argument('--tracker_threshold', default=150,
                        help='threshold between object detections and blob detections')
    parser.add_argument('--show_blob_update', default=True,
                        help='show when a blob is detected and accepted')

    args = parser.parse_args()

    return opt, args

if __name__ == "__main__":
    # Parse arg
    opt, args = arguments_parse()
    # run_centroid_tracker is no always a generator
    for img in run_blob_det_add_centroid_tracker(opt, args):
        print("IMAGE")




