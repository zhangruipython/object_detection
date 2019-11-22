# -*- coding: utf-8 -*-
"""
@Time    : 2019-11-22 9:43
@Author  : zhangrui
@FileName: parse_yolo.py
@Software: PyCharm
"""
import glob


def parseYoloFormat(file_path):
    bnd = open(file_path, "r+")
    bnd_line = open(file_path).readlines()
    for line in bnd_line:
        bnd.write(line.replace(",", ' '))
    bnd.close()


if __name__ == '__main__':
    label_path = "/home/hadoop/Documents/app/openimage/train/yike_train_data/"
    label_txt_list = glob.glob(label_path + "*.txt")
    for label_txt in label_txt_list:
        parseYoloFormat(label_txt)
