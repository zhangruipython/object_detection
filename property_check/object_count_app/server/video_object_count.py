# -*- coding: utf-8 -*-
"""
@Time    : 2019-12-23 11:13
@Author  : zhangrui
@FileName: video_object_count.py
@Software: PyCharm
视频类别盘点
"""
from deep_sort_module import nn_matching
from deep_sort_module.tracker import Tracker
from deep_sort_module.detection import Detection
from object_detect_module import do_detect, model_load
from tools import generate_detections
from deep_sort_module import preprocessing
from collections import Counter
import cv2
import numpy as np
import sys

sys.path.append('../../')
sys.path.append('../')


class DetectTrackCount:
    # 初始化deep_sort参数
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    model_filename = '/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/mars.pb'

    def __init__(self, video_path, frame_path, yolo_model_param, confidence):
        self.video_path = video_path
        self.frame_path = frame_path
        self.yolo_model_param = yolo_model_param
        self.confidence = confidence
        self.property_count = []

    def detect_track(self, zh_en_dir):
        encoder = generate_detections.create_box_encoder(self.model_filename, batch_size=1)
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

                label_boxes, label_names = do_detect.detection_to_tracker(detect_param["detections"], self.confidence,
                                                                          zh_en_dir)
                features = encoder(frame, label_boxes)
                detections = [Detection(bbox, 1.0, feature, label_name) for bbox, feature, label_name in
                              zip(label_boxes, features, label_names)]
                boxes = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                # 识别区非极大值抑制
                indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
                all_detections = [detections[i] for i in indices]
                # 执行跟踪操作
                tracker.predict()
                # 跟踪器更新，返回新识别类别名称
                update_label_names = tracker.update(all_detections)
                self.property_count += update_label_names
        video_capture.release()
        return self.property_count


if __name__ == '__main__':
    zh_en = {
        "1-1": "货架-货架1",
        "1-2": "货架-货架2",
        "1-3": "货架-货架3",
        "1-4": "货架-货架4",
        "1-5": "货架-货架5",
        "2-1": "椅子-高脚凳",
        "2-2": "椅子",
        "3": "结账机",
        "4-1": "桌子-餐桌",
        "5": "电子屏",
        "6-1": "器具柜-中岛柜",
        "6-2": "器具柜-风幕柜",
        "6-3": "器具柜-中心开放柜",
        "7-1": "冰箱-卧式冰柜",
        "7-2": "冰箱-四门冷藏柜",
        "7-3": "冰箱-单门冷藏柜",
        "7-4": "冰箱-双门冷藏柜",
        "7-5": "冰箱-二门冷藏工作台",
        "8-1": "电子设备-台式电脑",
        "9": "微波炉",
        "10-1": "汤锅-台上式单头多功能蒸煮汤锅",
        "11": "煎饼果子机",
        "12-1": "打印机-标签打印机",
        "12-2": "打印机-热敏打印机",
        "13-1": "电磁炉",
        "14-1": "扫描枪",
        "15-1": "POS机",
        "16-1": "电烤炉",
        "17-1": "面包展示柜-1",
        "17-2": "面包展示柜",
        "18-1": "试吃台",
        "19-1": "餐具-1",
        "19-2": "餐具-2",
        "20-1": "茶桶-不锈钢电热奶茶桶",
        "21-1": "榨汁机-1",
        "21-2": "榨汁机-2",
        "22-1": "冰淇淋机-三头桌上型冰激凌机",
        "23-1": "开水机-步进式开水机",
        "23-2": "开水机-蒸汽开水机",
        "24-1": "咖啡机-1",
        "25": "钱箱",
        "26-1": "电子秤"
    }
    # zh_en = {
    #     "person": "人",
    #     "bicycle": "自行车",
    #     "car": "汽车",
    #     "motorbike": "摩托车",
    #     "aeroplane": "飞机",
    #     "bus": "公交车",
    #     "train": "火车",
    #     "truck": "卡车",
    #     "boat": "船",
    #     "traffic light": "红绿灯",
    #     "fire hydrant ": "消防栓",
    #     "stop sign": "停止标志",
    #     "parking meter": "停车收费表",
    #     "bench": "板凳",
    #     "bird": "鸟",
    #     "cat": "猫",
    #     "dog": "狗",
    #     "horse": "马",
    #     "sheep": "羊",
    #     "cow": "牛",
    #     "elephant": "象",
    #     "bear": "熊",
    #     "zebra": "斑马",
    #     "giraffe": "长颈鹿",
    #     "backpack": "背包",
    #     "umbrella": "雨伞",
    #     "handbag": "手提包",
    #     "tie": "领带",
    #     "suitcase": "手提箱",
    #     "frisbee": "飞盘",
    #     "skis": "滑雪板",
    #     "snowboard": "单板滑雪",
    #     "sports ball": "运动球",
    #     "kite": "风筝",
    #     "baseball bat": "棒球棒",
    #     "baseball glove": "棒球手套",
    #     "skateboard": "滑板",
    #     "surfboard": "冲浪板",
    #     "tennis racket": "网球拍",
    #     "bottle": "瓶子",
    #     "wine glass": "红酒杯",
    #     "cup": "杯子",
    #     "fork": "叉子",
    #     "knife": "刀",
    #     "spoon": "勺",
    #     "bowl": "碗",
    #     "banana": "香蕉",
    #     "apple": "苹果",
    #     "sandwich": "三明治",
    #     "orange": "橙子",
    #     "broccoli": "西兰花",
    #     "carrot": "胡萝卜",
    #     "hot dog": "热狗",
    #     "pizza": "比萨",
    #     "donut": "甜甜圈",
    #     "cake": "蛋糕",
    #     "chair": "椅子",
    #     "sofa": "沙发",
    #     "pottedplant": "盆栽植物",
    #     "bed": "床",
    #     "diningtable": "餐桌",
    #     "toilet": "厕所",
    #     "tvmonitor": "电视监视器",
    #     "laptop": "笔记本电脑",
    #     "mouse": "鼠标",
    #     "remote": "远程",
    #     "keyboard": "键盘",
    #     "cell phone": "手机",
    #     "microwave": "微波炉",
    #     "oven": "烤箱",
    #     "toaster": "烤面包机",
    #     "sink": "水槽",
    #     "refrigerator": "冰箱",
    #     "book": "书",
    #     "clock": "时钟",
    #     "vase": "花瓶",
    #     "scissors": "剪刀",
    #     "teddy bear": "泰迪熊",
    #     "hair drier": "吹风机",
    #     "toothbrush": "牙刷",
    # }
    yolov3_model = model_load.load_yike_top(config_path="../../config.conf")
    detect_track_count = DetectTrackCount(video_path="/home/hadoop/Documents/darknet-master-1/darknet-master/test_file/cut7.mp4",
                                          frame_path="frame.jpg",
                                          yolo_model_param=yolov3_model, confidence=0.5)
    label_count = Counter(detect_track_count.detect_track(zh_en)).most_common()
    print(label_count)
