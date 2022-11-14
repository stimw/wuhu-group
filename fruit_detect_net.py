import sys
sys.path.insert(0, './net/yolov5')

from .yolov5.utils.datasets import LoadImages, LoadStreams, letterbox
from .yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from .yolov5.utils.torch_utils import select_device, time_synchronized
from .deep_sort_pytorch.utils.parser import get_config
from .deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

"""
class name for object, 12
"""
labels = ['apple', 'banana', 'orange', 'grape', 'eggplant', 'pear',
              'potato', 'tomato', 'broccoli', 'pepper', 'cherry','strawberry']


"""
translate (xin, ymin, xmax, ymax) to (x_center, y_center, width, height)
"""
def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, clses=None, offset=(0, 0)):
    """
    draw the result of detect and track in img
    """
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        #print('clses:', clses[i])
        cls  = clses[i] if clses is not None else 0
        #print('cls:', cls)
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)

        track_id  = '{}{:d}'.format("", id)
        txt = track_id +' '+ labels[int(cls[1])] + ' ' + '%.2f'%cls[0]
        #draw result
        t_size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        #print('cls:', cls)
        cv2.putText(img, txt, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


class DETECT_PARAM:
        IMG_SIZE = 640
        CONF_THRESH = 0.4
        IOU_THRESH = 0.5
        DEVICE = "3"

class DetectTracker():
    """
    class of algorithm
    """
    def __init__(self):
        #get parameter of detect from opt
        self.opt = get_opt()
        #get parameter of track from config
        cfg = get_config()
        cfg.merge_from_file(self.opt.config_deepsort)

        #define tracker
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

        # Initialize
        self.device = select_device(self.opt.device)
        self.half = True

        # Load model of detect
        self.model = torch.load(self.opt.weights, map_location=self.device)[
            'model'].float()  # load to FP32
        self.model.to(self.device).eval()
        if self.half:
            self.model.half()  # to FP16

    

        # Get names and colors
        #self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        # Run inference
        #t0 = time.time()
        img = torch.zeros((1, 3, DETECT_PARAM.IMG_SIZE, DETECT_PARAM.IMG_SIZE), device=self.device)  # init img
        # run once
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None

    def preprocess(self, inputs, img_size=416):
        """
        preprocess func for detect
        """

        # resize image to (640,640) 
        img = letterbox(inputs, new_shape=DETECT_PARAM.IMG_SIZE)[0]
        new_shape = img.shape
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img, new_shape
        
    def inference(self, im0):
        img, new_shape = self.preprocess(im0)

        # Inference
        pred = self.model(img)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes)

        for i, det in enumerate(pred):  # detections per image
           
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                bbox_xywh = []
                confs = []
                clss = []

                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item(), cls.item()])

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)
                
                # Pass detections to deepsort
                outputs = self.deepsort.update(xywhs, confss, im0)

                # draw boxes for visualization with res of tracker
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    # tracker_id
                    identities = outputs[:, -3]

                    # confidence and labels_id
                    cls_sing = outputs[:, -2:]
                    im0 = draw_boxes(im0, bbox_xyxy, identities, cls_sing)
               
            else:
                self.deepsort.increment_ages()
        return im0

    def clear(self):
        """
        Initialize all id of trackers
        """
        self.deepsort.tracker._next_id = 1
        self.deepsort.tracker.tracks.clear()

    

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='net/yolov5/weights/5m/best.pt', help='model.pt path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='inference/images', help='source')
    parser.add_argument('--output', type=str, default='inference/output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    # class 0-11 is labels
    parser.add_argument('--classes', nargs='+', type=int,
                        default=[0,1,2,3,4,5,6,7,8,9,10,11], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument("--config_deepsort", type=str,
                        default="net/deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    return args
    #test(args, 0)
    
