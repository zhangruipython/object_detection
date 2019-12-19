# -*- coding: utf-8 -*-
"""
@Time    : 2019-12-17 11:07
@Author  : zhangrui
@FileName: object_track.py
@Software: PyCharm
目标跟踪模块
"""
import cv2


class MultiTracker:
    def __init__(self, tracker_type):
        """
        :param tracker_type: 跟踪器类型
        """
        self.tracker_type = tracker_type

    # 根据选定跟踪器名称定义跟踪器
    def create_tracker(self):
        # 跟踪器类型
        trackerTypes = ["BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN", "MOSSE", "CSRT"]
        if self.tracker_type == trackerTypes[0]:
            tracker = cv2.TrackerBoosting_create()
        elif self.tracker_type == trackerTypes[1]:
            tracker = cv2.TrackerMIL_create()
        elif self.tracker_type == trackerTypes[2]:
            tracker = cv2.TrackerKCF_create()
        elif self.tracker_type == trackerTypes[3]:
            tracker = cv2.TrackerTLD_create()
        elif self.tracker_type == trackerTypes[4]:
            tracker = cv2.TrackerMedianFlow_create()
        elif self.tracker_type == trackerTypes[5]:
            tracker = cv2.TrackerGOTURN_create()
        elif self.tracker_type == trackerTypes[6]:
            tracker = cv2.TrackerMOSSE_create()
        elif self.tracker_type == trackerTypes[7]:
            tracker = cv2.TrackerCSRT_create()
        else:
            tracker = None
        return tracker

    def initialize_tracker(self, roi_boxes, initialize_frame):
        """
        多目标跟踪初始化
        :param roi_boxes: roi区域坐标集合[(label_name,(427, 392, 66, 58))]坐标说明：左上点x,左上点y,宽,高
        :param initialize_frame: 初始化图像
        :return:多目标跟踪器，跟踪器跟踪类别list
        """
        multiTracker = []
        for box in roi_boxes:
            tracker = self.create_tracker()
            tracker.init(initialize_frame, box[1])
            multiTracker.append(tracker)

        return multiTracker, roi_boxes

    @staticmethod
    def start_track(multi_tracker, track_labels, track_frame):
        """
        多目标开始跟踪
        :param multi_tracker:跟踪器list
        :param track_labels:跟踪器对应目标类别名称list
        :param track_frame:跟踪帧
        :return:跟踪数据 [(False, (0, 0), (0, 0), 'label01')]
        """
        track_box_param = []
        for i, tracker in enumerate(multi_tracker):
            success, box = tracker.update(track_frame)
            pt1 = (int(box[0]), int(box[1]))
            pt2 = (int(box[0] + box[2]), int(box[1] + box[3]))
            box_label_name = track_labels[i][0]
            track_box_param.append((success, pt1, pt2, box_label_name))
        return track_box_param
