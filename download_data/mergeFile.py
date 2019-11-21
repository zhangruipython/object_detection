# -*- coding: utf-8 -*-
"""
@Time    : 2019-11-19 17:33
@Author  : zhangrui
@FileName: mergeFile.py
@Software: PyCharm
复制多个文件夹中数据到单个文件夹
"""
import shutil
import glob

original_file_path = "/home/hadoop/Documents/app/openimage/image/train/Cake_stand/"
target_file_path = "/home/hadoop/Documents/app/openimage/image/train/all_data/"
original_file_list = glob.glob(original_file_path + "*")
for file_name in original_file_list:
    file = file_name.split(original_file_path)[1]
    target_file = target_file_path + file
    shutil.copy(file_name, target_file)

