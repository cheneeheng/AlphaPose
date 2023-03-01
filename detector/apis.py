# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Chao Xu (xuchao.19962007@sjtu.edu.cn)
# -----------------------------------------------------

"""API of detector"""
from abc import ABC, abstractmethod
import os


def get_detector(opt=None):
    if opt.detector == 'yolo':
        from detector.yolo_api import YOLODetector
        from detector.yolo_cfg import cfg
        if not os.path.exists(cfg.CONFIG):
            cfg.CONFIG = "submodules/AlphaPose/" + cfg.CONFIG
            cfg.WEIGHTS = "submodules/AlphaPose/" + cfg.WEIGHTS
        return YOLODetector(cfg, opt)
    elif 'yolox' in opt.detector:
        from detector.yolox_api import YOLOXDetector
        from detector.yolox_cfg import cfg
        if opt.detector.lower() == 'yolox':
            opt.detector = 'yolox-x'
        cfg.MODEL_NAME = opt.detector.lower()
        cfg.MODEL_WEIGHTS = f'detector/yolox/data/{opt.detector.lower().replace("-", "_")}.pth'
        if not os.path.exists(cfg.MODEL_WEIGHTS):
            cfg.MODEL_WEIGHTS = "submodules/AlphaPose/" + cfg.MODEL_WEIGHTS
        return YOLOXDetector(cfg, opt)
    elif opt.detector == 'tracker':
        from detector.tracker_api import Tracker
        from detector.tracker_cfg import cfg
        if not os.path.exists(cfg.CONFIG):
            cfg.CONFIG = "submodules/AlphaPose/" + cfg.CONFIG
            cfg.WEIGHTS = "submodules/AlphaPose/" + cfg.WEIGHTS
        return Tracker(cfg, opt)
    elif opt.detector.startswith('efficientdet_d'):
        from detector.effdet_api import EffDetDetector
        from detector.effdet_cfg import cfg
        return EffDetDetector(cfg, opt)
    else:
        raise NotImplementedError


class BaseDetector(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def image_preprocess(self, img_name):
        pass

    @abstractmethod
    def images_detection(self, imgs, orig_dim_list):
        pass

    @abstractmethod
    def detect_one_img(self, img_name):
        pass
