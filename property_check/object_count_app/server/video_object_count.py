# -*- coding: utf-8 -*-
"""
@Time    : 2019-12-23 11:13
@Author  : zhangrui
@FileName: video_object_count.py
@Software: PyCharm
视频类别盘点
针对多类别和多种类进行跟踪的思路：
1、yolo识别后按识别类别分类
2、针对每一种分类选用不同的外观匹配模型
3、对每一类别目标进行DeepSort跟踪预测
"""
from deep_sort_module import nn_matching
from deep_sort_module.tracker import Tracker
from deep_sort_module.detection import Detection
from object_detect_module import do_detect, model_load
from deep_sort_module import preprocessing
from object_count_app.server import store_appearance_model
from collections import Counter
from functools import reduce
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import sys
import time

sys.path.append('../../')
sys.path.append('../')

ProcessPool = ThreadPoolExecutor(max_workers=10)
# path_model_dict = {
#     "货架-货架1": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "货架-货架2": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "货架-货架3": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "货架-货架4": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "货架-货架5": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "椅子-高脚凳": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "椅子": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "结账机": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "桌子-餐桌": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "电子屏": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "器具柜-中岛柜": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "器具柜-风幕柜": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "器具柜-中心开放柜": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "冰箱-卧式冰柜": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "冰箱-四门冷藏柜": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "冰箱-单门冷藏柜": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "冰箱-双门冷藏柜": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "冰箱-二门冷藏工作台": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "电子设备-台式电脑": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "微波炉": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "汤锅-台上式单头多功能蒸煮汤锅": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "煎饼果子机": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "打印机-标签打印机": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "打印机-热敏打印机": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "电磁炉": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "扫描枪": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "POS机": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "电烤炉": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "面包展示柜-1": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "面包展示柜": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "试吃台": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "餐具-1": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "餐具-2": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "茶桶-不锈钢电热奶茶桶": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "榨汁机-1": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "榨汁机-2": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "冰淇淋机-三头桌上型冰激凌机": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "开水机-步进式开水机": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "开水机-蒸汽开水机": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "咖啡机-1": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "钱箱": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
#     "电子秤": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb"
# }
path_model_dict = {
    "轮胎": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
    "汽车": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
    "公共汽车": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
    "卡车": "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb",
}


class DetectTrackCount:
    # 初始化deep_sort参数
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    def __init__(self, video_path, frame_path, yolo_model_param, confidence):
        self.video_path = video_path
        self.frame_path = frame_path
        self.yolo_model_param = yolo_model_param
        self.confidence = confidence
        self.property_count = []
        self.store_model = store_appearance_model.StoreModel(path_model_dict=path_model_dict)

    def make_batch(self, label_param):
        label_box_list = [a[1] for a in label_param]
        label_name = label_param[0][0]
        encoder = self.store_model.get_model(label_name=label_name)
        label_patch = encoder(label_param[0][2], label_box_list).tolist()
        return label_patch

    def detect_track(self, zh_en_dir):
        # 外观匹配模型加载
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
        tracker = Tracker(metric)
        video_capture = cv2.VideoCapture(self.video_path)

        while True:
            ret, frame = video_capture.read()
            if not ret:
                video_capture.release()
                break
            else:
                cv2.imwrite(self.frame_path, frame)
                # yolo识别模块
                detect_param = do_detect.detection(self.yolo_model_param[3], self.yolo_model_param[0],
                                                   self.yolo_model_param[1], self.yolo_model_param[2],
                                                   self.frame_path)
                label_param = do_detect.detection_to_tracker(detect_param["detections"], self.confidence, zh_en_dir)
                # 按识别类别进行分类，一组类别对应一组跟踪器 [[('A', 0), ('A', 1)], [('C', 3)], [('B', 0), ('B', 2)]]
                label_values = set(map(lambda ws: ws[0], label_param))
                label_list = [[(y[0], y[1], frame) for y in label_param if y[0] == x] for x in label_values]

                if len(label_list) >= 2:
                    # 按识别类别进行外观模型匹配，通过map reduce提高速度
                    labels_batch_list = reduce(lambda x, y: x + y, list(ProcessPool.map(self.make_batch, label_list)))
                    label_boxes = reduce(lambda x, y: x + y, [[label[1] for label in labels] for labels in label_list])
                    label_names = reduce(lambda x, y: x + y, [[label[0] for label in labels] for labels in label_list])
                    features = np.array(labels_batch_list)
                    detections = [Detection(bbox, 1.0, feature, label_name) for bbox, feature, label_name in
                                  zip(label_boxes, features, label_names)]
                    boxes = np.array([d.tlwh for d in detections])
                    scores = np.array([d.confidence for d in detections])
                    # 识别区非极大值抑制
                    indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
                    all_detections = [detections[i] for i in indices]
                    # 执行跟踪操作
                    tracker.predict()
                    # 跟踪器更新，更新跟踪器list，返回新识别类别名称
                    update_label_names = tracker.update(all_detections)
                    self.property_count += update_label_names
        video_capture.release()
        return self.property_count


if __name__ == '__main__':
    zh_en = {
        "wheel": "轮胎",
        "car": "汽车",
        "bus": "公共汽车",
        "truck": "卡车",
        "night": "轮胎"
    }

    yolov3_model = model_load.load_wheel(config_path="../../config.conf")
    time01 = time.time()
    detect_track_count = DetectTrackCount(
        video_path="/home/hadoop/Documents/darknet-master-1/darknet-master/test_file/track_count.mp4",
        frame_path="frame.jpg",
        yolo_model_param=yolov3_model, confidence=0.5)
    label_count = Counter(detect_track_count.detect_track(zh_en)).most_common()
    print(label_count)
    print("耗时{a}".format(a=time.time() - time01))
