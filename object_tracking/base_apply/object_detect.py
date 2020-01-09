# -*- coding: utf-8 -*-
"""
@Time    : 2019-12-17 11:17
@Author  : zhangrui
@FileName: object_detect.py
@Software: PyCharm
目标识别
"""
import os
import sys
import cv2
import numpy
from PIL import ImageDraw, ImageFont, Image


def convertBack(x, y, w, h):
    x_min = (round(x - (w / 2.0)))
    x_max = (round(x + (w / 2.0)))
    y_min = (round(y - (h / 2.0)))
    y_max = (round(y + (h / 2.0)))
    return x_min, y_min, x_max, y_max


# 处理识别参数
def compute_detect_box(detections, box_image, zh_en):
    """
    将detect参数转化为roi box格式数据
    :param detections:
    :return:
    """
    detection_coo_list = []
    detection_pt_list = []
    if detections:
        for detection_param in detections:
            # TODO 添加置信度筛选，只选出置信度大于55%的识别区
            if detection_param[1] > 0.7:
                x, y, w, h = detection_param[2][0], detection_param[2][1], detection_param[2][2], detection_param[2][3]
                pt1_x, pt1_y, pt2_x, pt2_y = convertBack(x, y, w, h)
                detection_pt_list.append({"detect_label": zh_en[detection_param[0].decode()], "pt1": (pt1_x, pt1_y),
                                          "pt2": (pt2_x, pt2_y)})
                x_min, y_min, w, h = round(x - (w / 2.0)), round(y - (h / 2.0)), int(w), int(h)
                detection_coo_list.append((detection_param[0].decode(), (x_min, y_min, w, h)))
    return {"detection_coo_list": detection_coo_list, "detection_pt_list": detection_pt_list, "img": box_image}


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
    return [netMain, metaMain, thresh, darknet_path]


# 具体识别
def detection(darknet_path, net_main, meta_main, thresh, imagePath):
    sys.path.append(darknet_path)
    import darknet
    image_path = imagePath.encode("ascii")
    img = cv2.imread(imagePath)
    detections = darknet.detect(net_main, meta_main, image_path, thresh=thresh)
    return {"detections": detections, "img": img}


class DetectImage:
    def __init__(self, model_param, zh_en):
        self.model_param = model_param
        self.zh_en = zh_en

    def detect(self, image_path):
        detect_param = detection(self.model_param[3], self.model_param[0], self.model_param[1],
                                 self.model_param[2], image_path)
        detect_param_box = compute_detect_box(detections=detect_param["detections"], box_image=detect_param["img"],
                                              zh_en=self.zh_en)
        return detect_param_box


def draw_detection(detection_param):
    """
    矩形框框选识别区
    :param detection_param:识别参数
    :return:cv2图像对象
    """
    if detection_param["detection_pt_list"]:
        for param in detection_param["detection_pt_list"]:
            cv2.rectangle(detection_param["img"], param["pt1"], param["pt2"], (0, 255, 0), 3)
        pil_img = Image.fromarray(cv2.cvtColor(detection_param["img"], cv2.COLOR_RGB2BGR))
        img_w, img_h = detection_param["img"].shape[:2]
        if img_w > 3000:
            for param in detection_param["detection_pt_list"]:
                draw = ImageDraw.Draw(pil_img)
                front_style = ImageFont.truetype(font="/home/hadoop/Documents/SIMYOU.TTF", size=15, encoding="utf-8")
                draw.text((param["pt1"][0], param["pt1"][1] - 5), param["detect_label"], (255, 0, 0), front_style)
        else:
            for param in detection_param["detection_pt_list"]:
                draw = ImageDraw.Draw(pil_img)
                front_style = ImageFont.truetype(font="/home/hadoop/Documents/SIMYOU.TTF", size=15, encoding="utf-8")
                draw.text((param["pt1"][0], param["pt1"][1] - 5), param["detect_label"], (255, 0, 0), front_style)
        cv2_img = cv2.cvtColor(numpy.asarray(pil_img), cv2.COLOR_RGB2BGR)
        # cv2.imwrite(predicted_picture_name, cv2_img)
    else:
        # pil_img = Image.fromarray(cv2.cvtColor(detection_param["img"], cv2.COLOR_RGB2BGR))
        cv2_img = detection_param["img"]
    return cv2_img


if __name__ == '__main__':
    darknet_path = "/home/hadoop/Documents/darknet-master-1/darknet-master"
    configPath = "/home/hadoop/Documents/darknet-master-1/darknet-master/cfg/yolov3-yike02.cfg"
    weightPath = "/home/hadoop/Documents/darknet-master-1/darknet-master/backup/yolov3-yike02_final.weights"
    metaPath = "/home/hadoop/Documents/darknet-master-1/darknet-master/cfg/yike_train02.data"
    model_params = load_model(darknet_path, configPath, weightPath, metaPath)
    d_img = DetectImage(model_params)
    print(d_img.detect(image_path="/home/hadoop/Documents/darknet-master-1/darknet-master/test_file/demo01.png"))
