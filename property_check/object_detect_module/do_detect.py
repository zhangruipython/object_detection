# -*- coding: utf-8 -*-
"""
@Time    : 2019-11-26 13:26
@Author  : zhangrui
@FileName: do_detect.py
@Software: PyCharm
GPU darknet识别图像
"""
import os
import sys
import cv2
from PIL import Image, ImageDraw, ImageFont

netMain = None
metaMain = None
altNames = None


def performDetect(darknet_path, configPath, weightPath, metaPath, imagePath, thresh=0.25):
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
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents, re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception as e:
            print(e)
    # 判断图片路径是否存在
    if not os.path.exists(imagePath):
        raise ValueError("Invalid image path `" + os.path.abspath(imagePath) + "`")
    # 开始预测
    detections = darknet.detect(netMain, metaMain, imagePath.encode("ascii"), thresh)
    return detections


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
    frame_sized = cv2.imread(image_path)
    image_path = image_path.encode("ascii")
    detections = darknet.detect(net_main, meta_main, image_path, thresh=thresh)
    return {"detections": detections, "img": frame_sized}


def convertBack(x, y, w, h):
    x_min = (round(x - (w / 2.0)))
    x_max = (round(x + (w / 2.0)))
    y_min = (round(y - (h / 2.0)))
    y_max = (round(y + (h / 2.0)))
    return x_min, y_min, x_max, y_max


def convert(x, y, w, h):
    x_min = (round(x - (w / 2.0)))
    y_min = (round(y - (h / 2.0)))
    if x_min < 0:
        w = w + x_min
        x_min = 0
    if y_min < 0:
        h = h + y_min
        y_min = 0
    return x_min, y_min, int(w), int(h)


def draw_detection(detection_param):
    """
    矩形框框选识别区
    :param detection_param:识别参数
    :return:
    """
    if detection_param["detection_coo_list"]:
        for param in detection_param["detection_coo_list"]:
            cv2.rectangle(detection_param["img"], param["pt1"], param["pt2"], (0, 255, 0), 3)
        pil_img = Image.fromarray(cv2.cvtColor(detection_param["img"], cv2.COLOR_RGB2BGR))
        img_w, img_h = detection_param["img"].shape[:2]
        if img_w > 3000:
            for param in detection_param["detection_coo_list"]:
                draw = ImageDraw.Draw(pil_img)
                front_style = ImageFont.truetype(font="/home/hadoop/Documents/SIMYOU.TTF", size=120, encoding="utf-8")
                draw.text((param["pt1"][0], param["pt1"][1] - 5), param["detect_label"], (255, 0, 0), front_style)
        else:
            for param in detection_param["detection_coo_list"]:
                draw = ImageDraw.Draw(pil_img)
                front_style = ImageFont.truetype(font="/home/hadoop/Documents/SIMYOU.TTF", size=30, encoding="utf-8")
                draw.text((param["pt1"][0], param["pt1"][1] - 5), param["detect_label"], (255, 0, 0), front_style)
            # cv2_img = cv2.cvtColor(numpy.asarray(pil_img), cv2.COLOR_RGB2BGR)
            # cv2.imwrite(predicted_picture_name, cv2_img)
    else:
        pil_img = Image.fromarray(cv2.cvtColor(detection_param["img"], cv2.COLOR_RGB2BGR))
    return pil_img


def remove_close_element(boxes):
    """
    去除轴心距离相近的元素
    :param boxes:
    :return:
    """
    # close_threshold = 80
    sort_boxes = sorted(boxes, key=lambda a: a[1], reverse=False)
    b_list = []
    for i in range(len(sort_boxes) - 1):
        if abs(sort_boxes[i][1] - sort_boxes[i + 1][1]) < sort_boxes[1] * 0.2:
            if sort_boxes[i][2] < sort_boxes[i + 1][2]:
                b_list.append(sort_boxes[i])
            else:
                b_list.append(sort_boxes[i + 1])
    # 取出差集
    return list(set(sort_boxes).difference(set(b_list)))


def detection_to_tracker(detections, confidence, zh_en_dir):
    """
    将初始识别参数转为跟踪坐标参数
    :return:
    """
    label_param = []
    if detections:
        # 对应的标签名称，取出轴心x坐标，置信度，(top left x, top left y, width, height)
        detection_boxes_list = [(zh_en_dir[detection_param[0].decode()], detection_param[2][0], detection_param[1],
                                 convert(detection_param[2][0], detection_param[2][1], detection_param[2][2],
                                         detection_param[2][3]))
                                for detection_param in detections]

        # boxes = remove_close_element(detection_boxes_list)
        for detection_param in detection_boxes_list:
            # 置信度过滤
            if detection_param[2] > confidence:
                label_param.append([detection_param[0], detection_param[3]])
    return label_param
