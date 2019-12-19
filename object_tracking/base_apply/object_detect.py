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


def convertBack(x, y, w, h):
    x_min = (round(x - (w / 2.0)))
    x_max = (round(x + (w / 2.0)))
    y_min = (round(y - (h / 2.0)))
    y_max = (round(y + (h / 2.0)))
    return x_min, y_min, x_max, y_max


# 处理识别参数
def compute_detect_box(detections):
    """
    将detect参数转化为roi box格式数据
    :param detections:
    :return:
    """
    # zh_en_dir = {"Blender": "搅拌机",
    #              "Coffeemaker": "咖啡机",
    #              "Oven": "烤箱",
    #              "Microwave_oven": "微波炉",
    #              "Grinder": "磨豆机",
    #              "Chair": "椅子",
    #              "Refrigerator": "冰箱",
    #              "Cake_stand": "蛋糕架",
    #              "Wardrobe": "更衣柜",
    #              "Printer": "打印机",
    #              "Laptop": "笔记本电脑",
    #              "Tablet_computer": "平板电脑",
    #              "Camera": "摄像机",
    #              "Mobile_phone": "手机"}
    detection_coo_list = []
    if detections:
        for detection_param in detections:
            # TODO 添加置信度筛选，只选出置信度大于55%的识别区
            if detection_param[1] > 0.55:
                x, y, w, h = detection_param[2][0], detection_param[2][1], detection_param[2][2], detection_param[2][3]
                x_min, y_min, w, h = round(x - (w / 2.0)), round(y - (h / 2.0)), round(w), round(h)
                detection_coo_list.append((detection_param[0].decode(), (x_min, y_min, w, h)))
    else:
        detection_coo_list = []
    return {"detection_coo_list": detection_coo_list}


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
    detections = darknet.detect(net_main, meta_main, image_path, thresh=thresh)
    return {"detections": detections}


class DetectImage:
    def __init__(self, model_param):
        self.model_param = model_param

    def detect(self, image_path):
        detect_param = detection(self.model_param[3], self.model_param[0], self.model_param[1],
                                 self.model_param[2], image_path)
        detect_param_box = compute_detect_box(detections=detect_param["detections"])
        return detect_param_box


if __name__ == '__main__':
    darknet_path = "/home/hadoop/Documents/darknet-master-1/darknet-master"
    configPath = "/home/hadoop/Documents/darknet-master-1/darknet-master/cfg/yolov3-yike02.cfg"
    weightPath = "/home/hadoop/Documents/darknet-master-1/darknet-master/backup/yolov3-yike02_final.weights"
    metaPath = "/home/hadoop/Documents/darknet-master-1/darknet-master/cfg/yike_train02.data"
    model_params = load_model(darknet_path, configPath, weightPath, metaPath)
    d_img = DetectImage(model_params)
    print(d_img.detect(image_path="/home/hadoop/Documents/darknet-master-1/darknet-master/test_file/demo01.png"))
