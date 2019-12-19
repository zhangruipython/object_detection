# -*- coding: utf-8 -*-
"""
@Time    : 2019-12-17 10:51
@Author  : zhangrui
@FileName: track_count.py
@Software: PyCharm
"""
import cv2
from object_tracking.base_apply import object_track
from object_tracking.base_apply import object_detect
from collections import Counter


class TrackCount:
    def __init__(self, input_video_path, tracker_type, model_param, image_path):
        self.input_video_path = input_video_path
        self.tracker_type = tracker_type
        self.model_param = model_param
        self.image_path = image_path
        self.roi_trackers = [("start", (0, 0, 0, 0))]  # 跟踪器参数 [(label_name,(427, 392, 66, 58))] 坐标说明：左上点x,左上点y,宽,高
        self.object_num = []  # 识别类别list

    def update_tracker(self, detect_data_list, track_data_list):
        """
        对比跟踪数据和识别数据，更新跟踪器
        对比情况分为：
        1、识别数据和跟踪数据可以匹配上
        2、识别数据和跟踪数据无法匹配上
        :return: 跟踪参数
        """
        if detect_data_list:
            if track_data_list:
                track_label_list = [track_data[3] for track_data in track_data_list]
                for detect_data in detect_data_list:
                    detect_data_center = (detect_data[1][0] + detect_data[1][2]) / 2, \
                                         (detect_data[1][1] + detect_data[1][3]) / 2
                    # 如果识别类别中有跟踪器中不存在的类别，则添加该类别至跟踪器，且更新资产盘点数据
                    if detect_data[0] not in track_label_list:
                        self.roi_trackers.append(detect_data)
                        self.object_num.append(detect_data[0])
                    # 如果识别类别中存在和跟踪器类别相同的数据，比较box中心坐标偏差,若偏差在范围内则添加该类别至跟踪器，且更新资产盘点数据
                    for track_data in track_data_list:
                        track_data_center = (track_data[1][0] + track_data[2][0]) / 2, (
                                track_data[1][1] + track_data[2][1] / 2)
                        if detect_data[0] in track_label_list and \
                                abs(track_data_center[0] - detect_data_center[0]) <= 10 \
                                and abs(track_data_center[1] - detect_data_center[1] <= 10):
                            self.roi_trackers.append(detect_data)
                            self.object_num.append(detect_data[0])

            else:
                pass
        else:
            pass

    def read_video(self):
        video_cap = cv2.VideoCapture(self.input_video_path)
        multi_track = object_track.MultiTracker(self.tracker_type)
        # 初始化识别模块
        detect_image = object_detect.DetectImage(model_param=self.model_param)
        # 初始化跟踪模块
        success, start_frame = video_cap.read()
        initialize_tracker = multi_track.initialize_tracker(roi_boxes=self.roi_trackers, initialize_frame=start_frame)
        while video_cap.isOpened():
            success, frame = video_cap.read()
            if not success:
                video_cap.release()
                break
            else:
                # 开始识别，获取识别参数
                cv2.imwrite(self.image_path, frame)
                detect_param = detect_image.detect(image_path=self.image_path)
                print("识别参数{a}".format(a=detect_param["detection_coo_list"]))
                # 开始跟踪，获取跟踪参数 [(False, (0, 0), (0, 0), 'start')] 是否跟踪到，左上坐标，右下坐标，跟踪类别
                trackers_param = multi_track.start_track(multi_tracker=initialize_tracker[0],
                                                         track_labels=initialize_tracker[1], track_frame=frame)
                print("跟踪参数{b}".format(b=trackers_param))
                # 根据当前帧跟踪参数，更新跟踪器roi坐标至最新状态
                tracker_list = []
                for i, param in enumerate(trackers_param):
                    # 如果跟踪状态为true，则更新roi坐标
                    if param[0]:
                        tracker_list.append((param[3], (
                            param[1][0], param[1][1], (param[2][0] - param[1][0]), (param[2][1] - param[1][1]))))
                if tracker_list:
                    self.roi_trackers = tracker_list

                # # 根据是否跟踪成功，删除False跟踪器
                # for i, param in enumerate(trackers_param):
                #     if not param[0] and param[3] != "start":
                #         del trackers_param[i]
                # 根据识别数据更新跟踪参数
                self.update_tracker(detect_data_list=detect_param["detection_coo_list"], track_data_list=trackers_param)
                print("更新后识别参数{a}".format(a=self.roi_trackers))
                initialize_tracker = multi_track.initialize_tracker(roi_boxes=self.roi_trackers, initialize_frame=frame)
        video_cap.release()
        return self.object_num


if __name__ == '__main__':
    darknet_path = "/home/hadoop/Documents/darknet-master-1/darknet-master"
    configPath = "/home/hadoop/Documents/darknet-master-1/darknet-master/cfg/yolov3-yike02.cfg"
    weightPath = "/home/hadoop/Documents/darknet-master-1/darknet-master/backup/yolov3-yike02_final.weights"
    metaPath = "/home/hadoop/Documents/darknet-master-1/darknet-master/cfg/yike_train02.data"
    model_params = object_detect.load_model(darknet_path, configPath, weightPath, metaPath)
    track_count = TrackCount(
        input_video_path="/home/hadoop/Documents/darknet-master-1/darknet-master/test_file/cut2.mp4",
        tracker_type="MEDIANFLOW", model_param=model_params, image_path="frame.jpg")
    object_count = track_count.read_video()
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
        "18-1": "试吃台",
        "19-1": "餐具-1",
        "19-2": "餐具-2",
        "20-1": "茶桶-不锈钢电热奶茶桶",
        "21-1": "榨汁机-1",
        "21-2": "榨汁机-2",
        "22-1": "冰淇淋机-三头桌上型冰激凌机",
        "23-1": "开水机-步进式开水机",
        "23-2": "开水机-蒸汽开水机",
        "24-1": "咖啡机-1"
    }
    label_count = Counter([zh_en_dir[i]for i in object_count]).most_common()
    print(label_count)
