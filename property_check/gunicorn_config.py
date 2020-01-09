# -*- coding: utf-8 -*-
"""
@Time    : 2019-12-25 12:26
@Author  : zhangrui
@FileName: gunicorn_config.py
@Software: PyCharm
gunicorn配置
"""
import logging
import logging.handlers
from logging.handlers import WatchedFileHandler
import os
import multiprocessing

bind = "0.0.0.0:80"
backlog = 512  # 监听队列数量
worker_class = "gevent"
workers = 4  # 进程数
threads = 4  # 每个进程开启的线程数
timeout = 60  # 超时时间
loglevel = "info"  # 日志级别
access_log_format = '%(t)s %(p)s %(h)s "%(r)s" %(s)s %(L)s %(b)s %(f)s" "%(a)s"'
accesslog = "/home/hadoop/RongzerAI/rongerai/property_check/gunicorn_access.log"  # 访问日志文件
errorlog = "/home/hadoop/RongzerAI/rongerai/property_check/gunicorn_error.log"  # 错误日志文件
