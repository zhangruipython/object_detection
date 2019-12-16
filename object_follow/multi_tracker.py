# -*- coding: utf-8 -*-
"""
@Time    : 2019-12-06 17:40
@Author  : zhangrui
@FileName: multi_tracker.py
@Software: PyCharm
视频多目标跟踪
"""
import cv2.cv2 as cv2


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

    def start_track(self, roi_boxes, initialize_frame, video_frame):
        """
        开始跟踪（连续两帧跟踪区域计算相似度）
        :param roi_boxes: roi区域坐标集合[(label_name,(427, 392, 66, 58))]坐标说明：左上点x,左上点y,宽,高
        :param initialize_frame: 跟踪初始化视频帧
        :param video_frame: 跟踪计算视频帧
        :return: 跟踪帧所识别数据
        """
        # 创建Tracker对象
        tracker = self.create_tracker()
        # 初始化MultiTracker对象
        all_box_param = []
        for b_box in roi_boxes:
            # 初始化
            tracker.init(initialize_frame, b_box[1])
            # 单个类别跟踪数据 True (572.0, 279.0, 235.0, 201.0)
            success, box = tracker.update(video_frame)
            pt1 = (int(box[0]), int(box[1]))
            pt2 = (int(box[0] + box[2]), int(box[1] + box[3]))
            box_label_name = b_box[0]
            all_box_param.append((success, pt1, pt2, box_label_name))
        return all_box_param


if __name__ == '__main__':
    # MEDIANFLOW跟踪器
    tracker_name = "MEDIANFLOW"
    # picture01_path = "C:\\rongze\\data\\yike_picture\\video_picture\\0.jpg"
    # picture02_path = "C:\\rongze\\data\\yike_picture\\video_picture\\1.jpg"
    picture01_path = "C:\\rongze\\data\\yike_picture\\1\\1.jpg"
    picture02_path = "C:\\rongze\\data\\yike_picture\\1\\2.jpg"
    img01 = cv2.imread(picture01_path)
    img02 = cv2.imread(picture02_path)
    multi_track = MultiTracker(tracker_type=tracker_name)
    # [("label01", (246, 3, 1204, 593))]
    track_param = multi_track.start_track(roi_boxes=[], initialize_frame=img01,
                                          video_frame=img02)
    print(track_param)
