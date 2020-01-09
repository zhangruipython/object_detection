# -*- coding: utf-8 -*-
"""
@Time    : 2019-12-24 10:06
@Author  : zhangrui
@FileName: forms.py
@Software: PyCharm
"""
from django import forms


class UploadVideoForm(forms.Form):
    """
    视频表单上传
    """
    video_file = forms.FileField()
