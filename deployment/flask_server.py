import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
print("Parent dir", parentdir)
import random
import argparse
from os import listdir
from os.path import isfile, join
import natsort
import cv2
from flask import Flask, render_template, Response, redirect, url_for, request
from flask_bootstrap import Bootstrap
from tracking.tracker_centriod import run_centroid_tracker
from tracking.tracker_no import run_no_tracker
from datetime import datetime
import numpy as np

def arguments_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=parentdir+'/yolov5/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=parentdir+'/yolov5/data/bees_demo1.mp4', help='file, camera for webcam')
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
    parser.add_argument('--project', default=parentdir+'/yolov5/runs/detect', help='save results to project/name')
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
                        help='if tracking image results and images should be safed')
    parser.add_argument('--save_tracking_text', default=False,
                        help='if tracking text results and images should be safed')
    parser.add_argument('--show_tracking', default=False, help='view tracking')
    parser.add_argument('--show_trajectories', default=True, help='view tracking trajectories')
    parser.add_argument('--show_info', default=True, help='yield back img and info for flask')
    parser.add_argument('--yield_back', default=True, help='yield back img and info for flask')
    args = parser.parse_args()

    return opt, args

global args
global opt
opt, args = arguments_parse()
path_data = parentdir + "/yolov5/data/"
global file_path
global tracker_sel
tracker_sel = "NO tracker"
global tracker_info
track_info = {
    "time": [],
    "frame": [],
    "no_det_cur": [],
    "no_tr_cur": [],
    "ids_cur": [],
    "sum_tr": []
} # time, frame, no detec, no tracks, track ids

app = Flask(__name__)
bootstrap = Bootstrap(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/source', methods=['GET'])
def source():
    # get files in ./data/ to select and infere on
    file_list = [f for f in listdir(path_data) if isfile(join(path_data, f)) and (f[-3:] in ["mp4", "avi", "png", "jpg"])]
    print(file_list)
    file_list = (natsort.natsorted(file_list))
    print(file_list)
    file_list.append("Camera")
    print(file_list)
    return render_template('source.html', file_list=file_list)

@app.route('/source_selected', methods=['POST'])
def source_selected():
    global opt
    global file_path
    if request.form.get('file_select') != "Camera":
        file_path = path_data + request.form.get('file_select')
    else:
        file_path = request.form.get('file_select')
    opt.source = file_path
    print(file_path) # only file names
    if request.form.get('save_img') == "on":
        opt.nosave = False
    else:
        opt.nosave = True
    print(opt.nosave)
    if request.form.get('save_txt') == "on":
        opt.save_txt = True
    else:
        opt.save_txt = False
    print(opt.save_txt)
    if request.form.get('save_conf') == "on":
        opt.save_conf = True
    else:
        opt.save_conf = False
    print(opt.save_conf)
    opt.conf_thres = float(request.form.get('conf_thres'))/100
    print(opt.conf_thres)
    return redirect(url_for('tracker'))

@app.route('/tracker', methods=['GET'])
def tracker():
    tracker_list = ["Centriod", "OTHER", "NO tracker"]
    return render_template('tracker.html', tracker_list=tracker_list)

@app.route('/tracker_selected', methods=['POST'])
def tracker_selected():
    global tracker_sel
    tracker_sel = request.form.get('tracker_select')
    #print(tracker_sel)
    if tracker_sel == "Centriod":
        return redirect(url_for('centriod_tracker'))
    elif tracker_sel == "NO tracker":
        return redirect(url_for('inference'))
    return redirect(url_for('tracker'))

@app.route('/centriod_tracker', methods=['GET'])
def centriod_tracker():
    return render_template('centriod_tracker.html')

@app.route('/inference', methods=['GET', 'POST'])
def inference():
    global tracker_sel
    global args
    global track_info
    track_info = {
        "time": [],
        "frame": [],
        "no_det_cur": [],
        "no_tr_cur": [],
        "ids_cur": [],
        "sum_tr": []
    }
    if tracker_sel == "Centriod":
        maxDis = request.form["maxDisappeared"]
        args.maxDisappeared = int(maxDis)
        print(args.maxDisappeared)

        if request.form.get('show_trajectories') == "on":
            args.show_trajectories = True
        else:
            args.show_trajectories = False
        print(args.show_trajectories)
        if request.form.get('show_info') == "on":
            args.show_info = True
        else:
            args.show_info = False
        print(args.show_info)
        if request.form.get('save_tracking_img') == "on":
            args.save_tracking_img = True
        else:
            args.save_tracking_img = False
        print(args.save_tracking_img)
        if request.form.get('save_tracking_text') == "on":
            args.save_tracking_text = True
        else:
            args.save_tracking_text = False
        print(args.save_tracking_text)

    return render_template('inference.html')

@app.route('/documentation')
def documentation():
    return render_template('documentation.html')


def save_info_tracker(frame_no, no_det, no_tr, ids, sum_tr):
    """
    Save information
    :param frame_no:
    :param no_det:
    :param no_tr:
    :param ids:
    :param sum_tr:
    :return:
    """
    global track_info
    track_info["time"].append(datetime.now().strftime("%H:%M:%S"))
    track_info["frame"].append(frame_no)
    track_info["no_det_cur"].append(no_det)
    if no_tr is not None:
        track_info["no_tr_cur"].append(no_tr)
    if ids is not None:
        track_info["ids_cur"].append(ids)
    if ids is not None:
        track_info["sum_tr"].append(sum_tr)
    #print(track_info)


def info_tracker():
    global tracker_sel
    global args
    global opt

    if tracker_sel == "Centriod":
        for frame, no_det, no_tr, sum_tr, _, frame_no, ids in run_centroid_tracker(opt, args):
            save_info_tracker(frame_no, no_det, no_tr, ids, sum_tr)
            ret, buffer = cv2.imencode('.png', frame)
            img = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/png\r\n\r\n' + img + b'\r\n')
    elif tracker_sel == "NO tracker":
        for frame, no_det, _, frame_no in run_no_tracker(opt, args):
            save_info_tracker(frame_no, no_det, None, None, None)
            ret, buffer = cv2.imencode('.png', frame)
            img = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/png\r\n\r\n' + img + b'\r\n')

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(info_tracker(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/results', methods=['GET', 'POST'])
def results():
    global tracker_sel
    global track_info
    global args

    if tracker_sel == "Centriod":
        det_cur_g = np.array(track_info["no_det_cur"])
        frame_g = np.array(track_info["frame"]) # time if webcam
        tr_cur_g = np.array(track_info["no_tr_cur"])
        tr_sum_g = np.array(track_info["sum_tr"])
        time_cur_g = np.stack((frame_g, det_cur_g, tr_cur_g, tr_sum_g), axis=1)

        # For Candlestick graph
        ids_frame_list = []
        for id in range(0, track_info["sum_tr"][-1]):
            # create list to infer length of bee in the video
            ids_frame_list.append([id])
        for i, ids_frame in enumerate(track_info["ids_cur"]):
            # find all frames in which id is
            for id in ids_frame:
                ids_frame_list[id].append(i)
        for id in range(0, len(ids_frame_list)):
            # get first and last frame no.
            #print(id)
            if ids_frame_list[id][-1] != len(track_info["frame"]) - 1:
                # candle stick: # Bee Id, first frame, first frame, last frame no. - (if maxDis), last frame no.
                # so line will show the frames where it is still tracked
                ids_frame_list[id] = [ids_frame_list[id][0], ids_frame_list[id][1], ids_frame_list[id][1], (ids_frame_list[id][-1] - int(args.maxDisappeared)), ids_frame_list[id][-1]]
                # - max disaperance if not in last frame
            else:
                ids_frame_list[id] = [ids_frame_list[id][0], ids_frame_list[id][1], ids_frame_list[id][1],
                                      ids_frame_list[id][-1], ids_frame_list[id][-1]]
        #print(ids_frame_list)
        return render_template('results_track.html', time_det_cur=time_cur_g.tolist(), id_line=ids_frame_list)
    elif tracker_sel == "NO tracker":
        det_cur_g = np.array(track_info["no_det_cur"])
        frame_g = np.array(track_info["frame"])  # time if webcam
        time_cur_g = np.stack((frame_g, det_cur_g), axis=1)
        return render_template('results_notrack.html', time_det_cur=time_cur_g.tolist())

    return render_template('results_notrack.html')

if __name__ == '__main__':
    #app.run(debug=True, ssl_context='adhoc') # ssl_context='adhoc' for https https://blog.miguelgrinberg.com/post/running-your-flask-application-over-https

    app.run(host='0.0.0.0', debug=True, port=5000)



















