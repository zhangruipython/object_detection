# -*- coding: utf-8 -*-
"""
@Time    : 2019-12-24 21:32
@Author  : zhangrui
@FileName: demo01.py
@Software: PyCharm
"""

import os
DATA_URL_ROOT = "https://streamlit-self-driving.s3-us-west-2.amazonaws.com/"
a = os.path.join(DATA_URL_ROOT, "labels.csv.gz")
print(a)