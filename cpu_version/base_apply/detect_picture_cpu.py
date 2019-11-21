# -*- coding: utf-8 -*-
"""
@Time    : 2019-11-18 16:35
@Author  : zhangrui
@FileName: detect_picture_cpu.py
@Software: PyCharm
"""
import cv2
import numpy as np

# 初始化参数
confThreshold = 0.5
nmsThreshold = 0.4
inpWidth = 416
inpHeight = 416


# 获取输出层的名称
def getOutputsNames(net_work):
    layers_names = net_work.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layers_names[i[0] - 1] for i in net_work.getUnconnectedOutLayers()]


# 处理网络输出层
def post_process(img, net_outs, classes):
    img_height = img.shape[0]
    img_width = img.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    class_ids = []
    confidences = []
    boxes = []
    for out in net_outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confThreshold:
                center_x = int(detection[0] * img_width)
                center_y = int(detection[1] * img_height)
                width = int(detection[2] * img_width)
                height = int(detection[3] * img_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    # 识别类别参数
    detection_coo_list = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        # label_confidence = '%.2f' % confidences[i]
        label_name = classes[class_ids[i]]
        # detection_coo_list.append({"extract_label": label_name, "pt1": (top, width),
        #                            "pt2": ((left + width), (top + height))})
        detection_coo_list.append({"extract_label": label_name, "pt1": (left, top),
                                   "pt2": ((left + width), (top + height))})
    return {"detection_coo_list": detection_coo_list, "img_h": img_height, "img_w": img_width, "img": img}


def load_model(classes_file, model_configuration, model_weights):
    """
    模型预加载
    :param classes_file: names文件地址（标签类别名称）
    :param model_configuration: 模型cfg配置文件地址
    :param model_weights: 模型weights文件地址
    :return: 模型预加载文件
    """
    with open(classes_file, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    # 使用给出的模型配置文件和权重文件进行网络加载
    # 使用CPU
    cpu_net = cv2.dnn.readNetFromDarknet(model_configuration, model_weights)
    cpu_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    cpu_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return cpu_net, classes


def make_detect(model_net, my_classes, picture_path):
    """
    具体识别方法
    :param model_net:模型加载参数 （由加载模型方法传递）
    :param my_classes: 所有类别名称集合（由加载模型方法传递）
    :param picture_path: 识别图片地址
    :return: 图像识别参数
    """
    frame = cv2.imread(picture_path)
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
    model_net.setInput(blob)
    outs = model_net.forward(getOutputsNames(model_net))
    return post_process(img=frame, net_outs=outs, classes=my_classes)


def draw_detection(detection_param):
    """
    矩形框框选识别区
    :detections  预测图片名称
    return: 框选好的图片名称
    """
    predicted_picture_name = "../picture/prediction.jpg"
    for detection in detection_param["detection_coo_list"]:
        cv2.rectangle(detection_param["img"], detection["pt1"], detection["pt2"], (0, 255, 0), 1)
        cv2.putText(detection_param["img"], detection["extract_label"], (detection["pt1"][0], detection["pt1"][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], 2)
    cv2.imwrite(predicted_picture_name, detection_param["img"])


if __name__ == '__main__':
    names_path = "../data/coco.names"
    model_path = "../backup/yolov3.weights"
    cfg_path = "../cfg/yolov3.cfg"
    pic_path = "../picture/car01.jpg"
    model_param = load_model(classes_file=names_path, model_configuration=cfg_path, model_weights=model_path)
    param = make_detect(model_param[0], model_param[1], pic_path)
    print(param["detection_coo_list"])
    draw_detection(param)
