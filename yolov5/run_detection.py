
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


class Detector(object):
    @torch.no_grad()
    def __init__(self,
        weights='./best.pt',  # model.pt path(s)
        source='./data/bees_demo1.mp4',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        #mode="image", # mode of dataset #"video"
        bs=1, # batch_size
        ):
        #self.mode = None #"image"

        self.device = device
        self.half = half # DEPLYOMENT EFFICIENT
        self.augment = augment
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.max_det = max_det
        self.save_crop = save_crop
        self.save_conf = save_conf
        self.save_crop = save_crop
        self.view_img = view_img
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf
        self.line_thickness = line_thickness
        self.save_txt = save_txt
        self.update = update
        self.weights = weights

        self.save_img = not nosave and not source.endswith('.txt')  # save inference images
        self.webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

        # Directories
        self.save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (self.save_dir / 'labels' if save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        self.device = select_device(self.device)
        self.half &= self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        if half:
            self.model.half()  # to FP16

        self.vid_path, self.vid_writer = [None] * bs, [None] * bs

        # Second-stage classifier
        self.classify = False
        if self.classify:
            self.modelc = load_classifier(name='resnet50', n=2)  # initialize
            self.modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once

    @torch.no_grad()
    def run_detector_image(self, path, img, im0s, vid_cap, dataset):
        """
        Runs detection on a single image
        :param img: given image
        :return:
        """

        # Run inference
        t0 = time.time()

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = self.model(img, augment=self.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        t2 = time_synchronized()

        # Apply Classifier
        if self.classify:
            pred = apply_classifier(pred, self.modelc, img, im0s)

        res_list = []
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if self.webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(self.save_dir / p.name)  # img.jpg
            txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if self.save_crop else im0  # for save_crop

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results

                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                    if self.save_txt:  # Write to file
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    res_list.append(line)

                    if self.save_img or self.save_crop or self.view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=self.line_thickness)
                        if self.save_crop:
                            save_one_box(xyxy, imc, file=self.save_dir / 'crops' / self.names[c] / f'{p.stem}.jpg', BGR=True)
            print(res_list)
            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if self.view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if self.save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if self.vid_path[i] != save_path:  # new video
                        self.vid_path[i] = save_path
                        if isinstance(self.vid_writer[i], cv2.VideoWriter):
                            self.vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        self.vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    self.vid_writer[i].write(im0)

        if self.save_txt or self.save_img:
            s = f"\n{len(list(self.save_dir.glob('labels/*.txt')))} labels saved to {self.save_dir / 'labels'}" if self.save_txt else ''
            print(f"Results saved to {self.save_dir}{s}")

        if self.update:
            strip_optimizer(self.weights)  # update model (to fix SourceChangeWarning)

        print(f'Done. ({time.time() - t0:.3f}s)')

        return res_list, im0







