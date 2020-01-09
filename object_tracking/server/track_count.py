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
import numpy as np


class TrackCount:
    def __init__(self, input_video_path, tracker_type, model_param, image_path, output_video_path, zh_en):
        self.input_video_path = input_video_path
        self.tracker_type = tracker_type
        self.model_param = model_param
        self.image_path = image_path
        self.output_video_path = output_video_path
        self.zh_en = zh_en
        self.roi_trackers = [("start", (0, 0, 0, 0))]  # 跟踪器参数 [(label_name,(427, 392, 66, 58))] 坐标说明：左上点x,左上点y,宽,高
        self.object_num = []  # 识别类别list

    def update_tracker(self, detect_data_list, track_data_list):
        """
        对比跟踪数据和识别数据，更新跟踪器

        如果存在识别类别与跟踪类别类别名称系统且boxes坐标相似，则判定为同一物体，将其加入math_list
        最后通过对match_list与detect_list取差集，获取应该新增的跟踪类别

        :return: 跟踪参数
        """
        match_list = []
        if detect_data_list:
            if track_data_list:
                # track_label_list = [track_data[3] for track_data in track_data_list]
                for track_data in track_data_list:
                    tracker_center_box = [(track_data[1][0] + track_data[1][2] / 2),
                                          (track_data[1][1] + track_data[1][3] / 2)]
                    tracker_center_array = np.array(tracker_center_box)
                    for detect_data in detect_data_list:
                        detector_center_box = [(detect_data[1][0] + detect_data[1][2] / 2),
                                               (detect_data[1][1] + detect_data[1][3] / 2)]
                        detector_center_array = np.array(detector_center_box)
                        if detect_data[0] == track_data[0] and np.sqrt(
                                np.sum(np.square(detector_center_array - tracker_center_array))) <= 1 / 2 * \
                                detect_data[1][2]:
                            match_list.append(detect_data)

            else:
                pass
        else:
            pass

        un_match_list = (set(detect_data_list).difference(set(match_list)))
        # print(un_match_list)
        # self.roi_trackers += un_match_list
        self.object_num += [un_match_data[0] for un_match_data in un_match_list]

    def read_video(self):
        video_cap = cv2.VideoCapture(self.input_video_path)
        multi_track = object_track.MultiTracker(self.tracker_type)
        # 初始化识别模块
        detect_image = object_detect.DetectImage(model_param=self.model_param, zh_en=self.zh_en)
        # 初始化跟踪模块
        success, start_frame = video_cap.read()
        initialize_tracker = multi_track.initialize_tracker(roi_boxes=self.roi_trackers, initialize_frame=start_frame)
        # 获取视频参数
        frame_width = video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        video_fps = video_cap.get(cv2.CAP_PROP_FPS)
        four_cc = cv2.VideoWriter_fourcc(*'XVID')
        out_img = cv2.VideoWriter(self.output_video_path, four_cc, int(video_fps),
                                  (int(frame_width), int(frame_height)), True)

        while video_cap.isOpened():
            success, frame = video_cap.read()
            if not success:
                video_cap.release()
                break
            else:
                # 开始识别，获取识别参数
                cv2.imwrite(self.image_path, frame)
                detect_param = detect_image.detect(image_path=self.image_path)
                # print("识别参数{a}".format(a=detect_param["detection_coo_list"]))

                # 图像识别框和识别类别绘制
                cv2_img = object_detect.draw_detection(detect_param)
                out_img.write(cv2_img)
                # 开始跟踪，获取跟踪参数 [(False, (0, 0), (0, 0), 'start')] 是否跟踪到，左上坐标，右下坐标，跟踪类别
                trackers_param = multi_track.start_track(multi_tracker=initialize_tracker[0],
                                                         track_labels=initialize_tracker[1], track_frame=frame)
                # 根据当前帧跟踪参数，更新跟踪器roi坐标至最新状态
                tracker_list = []

                for i, param in enumerate(trackers_param):
                    # 如果跟踪状态为true，则更新roi坐标
                    if param[0]:
                        tracker_list.append((param[3], (
                            param[1][0], param[1][1], (param[2][0] - param[1][0]), (param[2][1] - param[1][1]))))

                # 在roi_trackers中添加新识别类别
                self.update_tracker(detect_data_list=detect_param["detection_coo_list"],
                                    track_data_list=tracker_list)
                self.roi_trackers = detect_param["detection_coo_list"]
                print("跟踪参数{a}".format(a=self.roi_trackers))
                initialize_tracker = multi_track.initialize_tracker(roi_boxes=self.roi_trackers, initialize_frame=frame)
        video_cap.release()
        out_img.release()
        return self.object_num


if __name__ == '__main__':
    # darknet_path = "/home/hadoop/Documents/darknet-master-1/darknet-master/"
    # configPath = "/home/hadoop/Documents/darknet-master-1/darknet-master/cfg/yolov3-yike02.cfg"
    # weightPath = "/home/hadoop/Documents/darknet-master-1/darknet-master/final_backup/yolov3-yike02_final.weights"
    # metaPath = "/home/hadoop/Documents/darknet-master-1/darknet-master/cfg/yike_train02.data"
    darknet_path = "/home/hadoop/Documents/darknet-master-1/darknet-master"
    config_path = "/home/hadoop/Documents/darknet-master-1/darknet-master/cfg/yolov3-yike_top.cfg"
    weight_path = "/home/hadoop/Documents/darknet-master-1/darknet-master/backup/yolov3-yike_top_final.weights"
    meta_path = "/home/hadoop/Documents/darknet-master-1/darknet-master/cfg/yike_top.data"
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
    # zh_en_dir = {
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
    model_params = object_detect.load_model(darknet_path, config_path, weight_path, meta_path)
    track_count = TrackCount(
        input_video_path="/home/hadoop/Documents/darknet-master-1/darknet-master/test_file/cut7.mp4",
        tracker_type="MEDIANFLOW", model_param=model_params, image_path="frame.jpg",
        output_video_path="/home/hadoop/Documents/darknet-master-1/darknet-master/test_file/out.avi",
        zh_en=zh_en_dir)
    object_count = track_count.read_video()
    label_count = Counter([zh_en_dir[i] for i in object_count]).most_common()
    print(label_count)
