# -*- coding: utf-8 -*-
"""
@Time    : 2019-12-09 18:05
@Author  : zhangrui
@FileName: demo02.py
@Software: PyCharm
"""
import cv2
import os
from collections import Counter

print(os.cpu_count())


def fun01(list01):
    return list01[0] + list01[1]


def group_two(layer_list):
    """
        将list中元素两两划分
        :param layer_list:传入list
        :return: 二维list
        """
    return [layer_list[i:i + 2] for i in range(0, len(layer_list), 2)]


# b = list(map(fun01, [[2, 3]]))
# print(group_two(layer_list=[1, 2, 3, 4, 5, 6]))
# img_src = 'http://n1image.hjfile.cn/shetuan/2017-05-17-1495016837-986-732.jpg'
# cap = cv2.VideoCapture(img_src)
# if cap.isOpened():
#     ret, img = cap.read()
#     cv2.imwrite("a.jpg", img)
#     # cv2.waitKey(1)
list01 = [{"label": 1}, {"label": 1}, {"label": 1}, {"label": 1}, {"label": 2}, {"label": 2}]
list02 = [i["label"] for i in list01]
count = Counter(list02).most_common()
list03 = []
for b in count:
    list03.append({"label": b[0], "count": b[1]})
# print(list03)
img = cv2.imread("C:\\rongze\\data\\yike_picture\\demo_mix.jpg")
a = cv2.copyMakeBorder(img, 0, 0, 100, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
cv2.imwrite("C:\\rongze\\data\\yike_picture\\demo_mix.jpg", a)
