# -*- coding: utf-8 -*-
"""
@Time    : 2019-12-24 10:54
@Author  : zhangrui
@FileName: urls.py
@Software: PyCharm
"""
from django.conf.urls import url
from object_count_app.views import ObjectCountView

urlpatterns = [
    url(r'^object_count$', ObjectCountView.as_view()),
]
