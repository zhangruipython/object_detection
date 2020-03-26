# -*- coding: utf-8 -*-
"""
@Time    : 2020-01-14 10:10
@Author  : zhangrui
@FileName: store_appearance_model.py
@Software: PyCharm
"""
from tools import generate_detections


class StoreModel:
    def __init__(self, path_model_dict):
        """
        :param path_model_dict: 外观匹配模型地址和外观名称字典
        """
        self.path_model_dict = path_model_dict
        # 外观匹配模型实例化类与外观名称字典映射
        self.label_model_dict = {}

    def set_model(self, label_name):
        model_file_path = self.path_model_dict[label_name]
        encoder = generate_detections.create_box_encoder(model_filename=model_file_path, batch_size=1)
        self.label_model_dict[label_name] = encoder

    def get_model(self, label_name):
        """
        根据类别名称查询对应的外观模型类，如果在字典的key中存在该类别名称则直接返回结果，
        如果不存在则做模型加载，且存储在字典中
        :param label_name:
        :return:
        """
        if label_name in self.label_model_dict.keys():
            return self.label_model_dict[label_name]
        else:
            self.set_model(label_name)
            return self.label_model_dict[label_name]