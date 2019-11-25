# -*- coding: utf-8 -*-
"""
@Time    : 2019-11-19 15:37
@Author  : zhangrui
@FileName: splitTrainAndTest.py
@Software: PyCharm
区分训练集和测试集
"""
import random
import os


def split_data_set(image_dir):
    f_val = open("snowman_test.txt", 'w')
    f_train = open("snowman_train.txt", 'w')

    path, dirs, files = next(os.walk(image_dir))
    data_size = len(files)

    ind = 0
    # 1/10的数据集作为测试集
    data_test_size = int(0.1 * data_size)
    test_array = random.sample(range(data_size), k=data_test_size)
    for f in os.listdir(image_dir):
        if f.split(".")[1] == "jpg":
            ind += 1
            if ind in test_array:
                f_val.write(image_dir + '/' + f + '\n')
            else:
                f_train.write(image_dir + '/' + f + '\n')


data_path = "C:\\Users\\张睿\\Desktop\\ImgBiao"
split_data_set(data_path)
