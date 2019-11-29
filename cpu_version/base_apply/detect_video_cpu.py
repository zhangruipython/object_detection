# -*- coding: utf-8 -*-
"""
@Time    : 2019-11-22 16:45
@Author  : zhangrui
@FileName: detect_video_cpu.py
@Software: PyCharm
解析视频，识别视频中每一帧
"""
from cpu_version.base_apply import detect_picture_cpu
import cv2
frame_write_path = "C:/rongze/picture/car/tyre/"  # 视频帧写入文件
video_path = "C:/rongze/picture/car/big_car03.mp4"  # 视频地址
frame_index = 0  # 视频帧文件名称

names_path = "../data/coco.names"
model_path = "../backup/yolov3.weights"
cfg_path = "../cfg/yolov3.cfg"
model_param = detect_picture_cpu.load_model(classes_file=names_path, model_configuration=cfg_path,
                                            model_weights=model_path)

capture = cv2.VideoCapture(video_path)
while capture.isOpened():
    ret, frame = capture.read()
    if frame is None:
        break
    else:
        frame_write_name = frame_write_path + str(frame_index) + ".jpg"
        cv2.imwrite(frame_write_name, frame)
        frame_index += 1
        param = detect_picture_cpu.make_detect(model_param[0], model_param[1], frame_write_name)
        detect_picture_cpu.draw_detection(param)
capture.release()
