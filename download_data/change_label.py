# -*- coding: utf-8 -*-
"""
@Time    : 2019-11-21 9:25
@Author  : zhangrui
@FileName: change_label.py
@Software: PyCharm
将标签名称改为对应list中索引
classes = ["Blender", "Coffeemaker", "Oven", "Microwave_oven", "Grinder", "Chair", "Refrigerator", "Cake_stand",
           "Wardrobe", "Printer", "Laptop", "Tablet_computer", "Camera", "Mobile_phone"]
{Microwave_oven:3,
Cake_stand:7,
Tablet_computer:11,
Mobile_phone:13}
names 13
"""
import glob


def change_label_name(label_name, label_index):
    label_path = "/home/hadoop/Documents/app/openimage/image/train/{a}/".format(a=label_name)
    class_index = label_index
    label_txt_list = glob.glob(label_path + "*.txt")
    for label_txt in label_txt_list:
        line_label = open(label_txt, "r+")
        for line in open(label_txt):
            line_label.write(line.replace(label_name, class_index))
        line_label.close()


label_dir = {"Microwave_oven": "3",
             "Cake_stand": "7",
             "Tablet_computer": "11",
             "Mobile_phone": "13"}
for label in label_dir:
    change_label_name(label_name=label, label_index=label_dir[label])
