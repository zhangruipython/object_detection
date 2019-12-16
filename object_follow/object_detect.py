# -*- coding: utf-8 -*-
"""
@Time    : 2019-12-10 11:22
@Author  : zhangrui
@FileName: object_detect.py
@Software: PyCharm
目标识别（识别区去重）
"""
import sys
import os
import cv2


# 加载模型[netMain, metaMain, thresh]
def load_model(darknet_path, configPath, weightPath, metaPath, thresh=0.25):
    sys.path.append(darknet_path)
    import darknet
    global metaMain, netMain, altNames
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" + os.path.abspath(configPath) + "`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" + os.path.abspath(weightPath) + "`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" + os.path.abspath(metaPath) + "`")

    # 默认batch为1
    netMain = darknet.load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)
    metaMain = darknet.load_meta(metaPath.encode("ascii"))
    return netMain, metaMain, thresh, darknet_path


# 具体识别
def detection(darknet_path, net_main, meta_main, thresh, image_path):
    """
    通过传入模型预加载参数进行图片目标识别
    :param darknet_path:darknet.py文件所在地址
    :param net_main:模型加载文件
    :param meta_main:模型加载文件
    :param thresh:阈值
    :param image_path:识别图像地址
    :return:{"detections": detections, "img": 图像矩阵}
    如果不存在识别区detections为[]
    """
    sys.path.append(darknet_path)
    import darknet
    print("图片地址{a}".format(a=image_path))
    print(type(image_path))
    frame_sized = cv2.imread(image_path)
    image_path = image_path.encode("ascii")
    detections = darknet.detect(net_main, meta_main, image_path, thresh=thresh)
    return {"detections": detections, "img": frame_sized}
