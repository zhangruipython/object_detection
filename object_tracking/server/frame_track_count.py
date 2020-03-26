# -*- coding: utf-8 -*-
"""
@Time    : 2020-01-20 13:36
@Author  : zhangrui
@FileName: frame_track_count.py
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
                print("识别参数{a}".format(a=detect_param["detection_coo_list"]))

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
    darknet_path = "/home/hadoop/Documents/darknet-master-1/darknet-master"
    config_path = "/home/hadoop/Documents/darknet-master-1/darknet-master/cfg/yolov3-all_train.cfg"
    weight_path = "/home/hadoop/Documents/darknet-master-1/darknet-master/backup/yolov3-all_train_final.weights"
    meta_path = "/home/hadoop/Documents/darknet-master-1/darknet-master/cfg/all_train.data"
    zh_en_dir = {
        "wheel": "轮胎",
        "car": "汽车",
        "bus": "公共汽车",
        "truck": "卡车",
        "night": "轮胎"
    }
    model_params = object_detect.load_model(darknet_path, config_path, weight_path, meta_path)
    track_count = TrackCount(
        input_video_path="/home/hadoop/Documents/darknet-master-1/darknet-master/test_file/big_car02.mp4",
        tracker_type="MEDIANFLOW", model_param=model_params, image_path="frame.jpg",
        output_video_path="/home/hadoop/Documents/darknet-master-1/darknet-master/test_file/out.avi",
        zh_en=zh_en_dir)
    object_count = track_count.read_video()
    label_count = Counter([zh_en_dir[i] for i in object_count]).most_common()
    print(label_count)
