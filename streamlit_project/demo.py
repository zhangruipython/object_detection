# -*- coding: utf-8 -*-
"""
@Time    : 2019-12-24 19:14
@Author  : zhangrui
@FileName: demo.py
@Software: PyCharm
"""
import streamlit as st
import numpy as np
import pandas as pd
import time
st.title("第一个streamlit应用")
# st.write(pd.DataFrame({'first column': [1, 2, 3, 4], 'second column': [10, 20, 30, 40]}))
# 使用复选框
if st.checkbox("展示地图"):
    df = pd.DataFrame(
        np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
        columns=['lat', 'lon'])
    st.map(df)
elif st.checkbox("展示折线图"):
    data = pd.DataFrame({"first column": [1, 2, 3, 4], "second column": [10, 20, 30, 40]})
    st.line_chart(data)
df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
})
# 侧边栏选择框
option = st.sidebar.selectbox('Which number do you like best?', df["first column"])
"所选择", option

# 进度条
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(10):
    latest_iteration.text(f'Iteration {i+1}')
    bar.progress(i + 1)
    time.sleep(0.1)
'...and now we\'re done!'
