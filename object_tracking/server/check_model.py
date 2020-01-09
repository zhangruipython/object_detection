# -*- coding: utf-8 -*-
"""
@Time    : 2020-01-08 13:16
@Author  : zhangrui
@FileName: check_model.py
@Software: PyCharm
模型检验，找到视频中不能识别出指定类别的帧
"""
from object_tracking.base_apply import object_detect
import cv2
import uuid


class CheckModel:
    def __init__(self, input_video_path, model_param, un_detect_path, image_path, zh_en):
        self.input_video_path = input_video_path
        self.model_param = model_param
        self.un_detect_path = un_detect_path
        self.image_path = image_path
        self.zh_en = zh_en

    def make_check(self, check_label):
        detect_image = object_detect.DetectImage(model_param=self.model_param, zh_en=self.zh_en)
        video_cap = cv2.VideoCapture(self.input_video_path)
        while video_cap.isOpened():
            success, frame = video_cap.read()
            if not success:
                video_cap.release()
                break
            else:
                # 开始识别，获取识别参数
                cv2.imwrite(self.image_path, frame)
                detect_param = detect_image.detect(image_path=self.image_path)
                detect_labels = [param[0] for param in detect_param["detection_coo_list"]]
                print(detect_labels)
                if check_label not in detect_labels:
                    cv2.imwrite(self.un_detect_path + str(uuid.uuid4()) + ".jpg", frame)
        video_cap.release()


if __name__ == '__main__':
    darknet_path = "/home/hadoop/Documents/darknet-master-1/darknet-master"
    config_path = "/home/hadoop/Documents/darknet-master-1/darknet-master/cfg/yolov3-yike_top.cfg"
    weight_path = "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/yolov3-yike_top_final.weights"
    meta_path = "/home/hadoop/Documents/darknet-master-1/darknet-master/cfg/yike_top.data"
    model_params = object_detect.load_model(darknet_path, config_path, weight_path, meta_path)
    zh_en_dir = {
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
    check_model = CheckModel(
        input_video_path="/home/hadoop/Documents/darknet-master-1/darknet-master/test_file/cut7.mp4",
        model_param=model_params, image_path="frame.jpg",
        un_detect_path="/home/hadoop/Documents/darknet-master-1/darknet-master/test_file/un_detect/",
        zh_en=zh_en_dir)
    check_model.make_check(check_label="21-1")
