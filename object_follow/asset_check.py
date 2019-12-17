# -*- coding: utf-8 -*-
"""
@Time    : 2019-12-10 13:20
@Author  : zhangrui
@FileName: asset_check.py
@Software: PyCharm
视频识别跟踪
"""
from object_follow.object_detect import detection
from object_follow.object_detect import load_model
from object_follow.multi_tracker import MultiTracker
import cv2

asset_video_path = ""
frame_pic_path = ""
asset_frame_path = ""

asset_cap = cv2.VideoCapture(asset_video_path)

darknet_path = ""
configPath = ""
weightPath = ""
metaPath = ""
model_param = load_model(darknet_path, configPath, weightPath, metaPath)


def detect_cam():
    while asset_cap.isOpened():
        success, frame = asset_cap.read()
        if not success:
            print("视频流读取失败")
            break
        else:
            # 开始识别
            cv2.imwrite(frame_pic_path, frame)
            detect_param = detection(model_param[3], model_param[0], model_param[1], model_param[2], frame_pic_path)
            yield detect_param


def get_detect_box(detections, img):
    """
    处理识别参数
    :param img: 图像
    :param detections: 初始识别数据
    :return: 处理后参数
    """
    detection_coo_list = []
    if detections:
        for detection_param in detections:
            # TODO 添加置信度筛选，只选出置信度大于55%的识别区
            if detection_param[1] > 0.55:
                x, y, w, h = detection_param[2][0], detection_param[2][1], detection_param[2][2], detection_param[2][3]
                x_min, y_min = round(x - (w / 2.0)), round(y - (h / 2.0))
                detection_coo_list.append((detection_param[0].decode(), (x_min, y_min, w, h)))
    else:
        detection_coo_list = []
    return {"detection_coo_list": detection_coo_list, "img": img}


def comparison_param(detect_param, track_param):
    """
    比对识别参数和跟踪参数

    1、判断是否将识别类别添加至盘点表中
    （1）识别数据中出现跟踪数据没有出现过的类别
    （2）识别数据中出现与跟踪数据坐标偏差较大的类别
    2、判断下一轮跟踪目标
    （1）识别数据中出现跟踪数据中没有出现过的类别
    （2）识别数据中识别类别为true的类别
    :param detect_param: 识别参数
    :param track_param: 跟踪参数
    :return:下一轮跟踪器跟踪对象
    """
    # 新增盘点数据
    add_data = []
    for track_coo in track_param:
        track_label = track_coo[3]
        track_coo_center = ((track_coo[1][0] + track_coo[2][0]) / 2, (track_coo[1][1] + track_coo[2][1]) / 2)
        for detection_coo in detect_param["detection_coo_list"]:
            detect_label = detection_coo[0]
            x_min, y_min, w, h = detection_coo[1]
            detect_box_center = (x_min + (w / 2), y_min + (h / 2))
            # 识别数据中出现跟踪数据没有出现过的类别
            if detect_label != track_label:
                add_data.append(detect_label)
            # 识别数据中出现与跟踪数据坐标偏差较大的类别，按中心坐标偏差计算
            elif detect_label == track_label and track_coo_center[0] - detect_box_center[0] > 10 and track_coo_center[
                1] - detect_box_center[1] > 5:
                add_data.append(detect_label)

    # 跟踪器添加类别
    track_data_list = []
    for track_coo in track_param:
        track_label = track_coo[3]
        if track_coo[0]:
            track_box = (track_coo[3], (track_coo[1][0], track_coo[1][1], track_coo[2][0] - track_coo[1][0],
                                        track_coo[2][1] - track_coo[1][1]))
            track_data_list.append(track_box)
        for detection_coo in detect_param["detection_coo_list"]:
            detect_label = detection_coo[0]
            if detect_label != track_label:
                track_data_list.append(detection_coo)
    return track_data_list


def handle_param():
    """
    处理识别数据和跟踪数据
    :return:
    """
    multi_tracker = MultiTracker(tracker_type="MEDIANFLOW")
    # 长度为2的识别数据list
    detect_param_list = []
    # next跟踪参数
    track_roi_boxes = []
    for detect_param in detect_cam():
        # 将识别数据转化为[(label_name,(427, 392, 66, 58))]坐标说明：左上点x,左上点y,宽,高
        detect_box_param = get_detect_box(detect_param["detections"], detect_param["img"])
        detect_param_list.append(detect_box_param)

        if len(detect_param_list) == 2:
            # 如果list中存在两张图片，则开始跟踪 跟踪返回参数 [(True, (246, 9), (1449, 601), 'label01')]
            # detect_param_list[0]["detection_coo_list"]
            track_param = multi_tracker.start_track(roi_boxes=track_roi_boxes,
                                                    initialize_frame=detect_param_list[0]["img"],
                                                    video_frame=detect_param_list[1]["img"])
            # 对比跟踪参数和识别参数，获取下一轮跟踪参数
            track_roi_boxes = comparison_param(detect_param, track_param)
            detect_param_list = []
