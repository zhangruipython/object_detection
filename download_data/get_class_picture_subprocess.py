# -*- coding: utf-8 -*-
"""
@Time    : 2019-11-19 15:21
@Author  : zhangrui
@FileName: downloadIO.py
@Software: PyCharm
参考链接：https://www.learnopencv.com/fast-image-downloader-for-open-images-v4/
多进程下载OpenImagesV4数据集
使用案例：python3 downloadIO.py --classes 'Ice_cream,table' --mode train
"""
import argparse
import csv
import subprocess
import os
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool as thread_pool
import pandas as pd
cpu_count = multiprocessing.cpu_count()

parser = argparse.ArgumentParser(description='Download Class specific images from OpenImagesV4')
parser.add_argument("--mode", help="Dataset category - train, validation or test", required=True)
# parser.add_argument("--classes", help="Names of object classes to be downloaded", required=True)
parser.add_argument("--nthreads", help="Number of threads to use", required=False, type=int, default=cpu_count * 2)
parser.add_argument("--occluded", help="Include occluded images", required=False, type=int, default=1)
parser.add_argument("--truncated", help="Include truncated images", required=False, type=int, default=1)
parser.add_argument("--groupOf", help="Include groupOf images", required=False, type=int, default=1)
parser.add_argument("--depiction", help="Include depiction images", required=False, type=int, default=1)
parser.add_argument("--inside", help="Include inside images", required=False, type=int, default=1)

args = parser.parse_args()

run_mode = args.mode

threads = args.nthreads

# classes = ["Blender", "Coffeemaker"]
# 读取cvs中标签名称
csv_path = "/home/hadoop/Documents/openimage/image/class-descriptions-boxable.csv"
classes = pd.read_csv(csv_path)["Tortoise"].tolist()
# for class_name in args.classes.split(','):
#     classes.append(class_name)

with open('./class-descriptions-boxable.csv', mode='r') as infile:
    reader = csv.reader(infile)
    dict_list = {rows[1]: rows[0] for rows in reader}

subprocess.run(['rm', '-rf', run_mode])
subprocess.run(['mkdir', run_mode])
subprocess.run(['rm', '-rf', 'labels'])
subprocess.run(['mkdir', 'labels'])
pool = thread_pool(threads)
commands = []
cnt = 0

for ind in range(0, len(classes)):

    className = classes[ind]
    print("Class " + str(ind) + " : " + className)

    commandStr = "grep " + dict_list[className] + " " + run_mode + "-annotations-bbox.csv"
    print(commandStr)
    class_annotations = subprocess.run(commandStr.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')
    class_annotations = class_annotations.splitlines()

    totalNumOfAnnotations = len(class_annotations)
    print("Total number of annotations : " + str(totalNumOfAnnotations))
    cnt = 0
    for line in class_annotations[0:totalNumOfAnnotations]:
        cnt = cnt + 1
        print("annotation count : " + str(cnt))
        lineParts = line.split(',')
        command = 'aws s3 --no-sign-request --only-show-errors cp s3://open-images-dataset/' + run_mode + '/' + lineParts[0] + ".jpg"+'JPEGImages/' + lineParts[0] + ".jpg"
        commands.append(command)
        with open('labels/%s.txt' % (lineParts[0]), 'a') as f:
            f.write(' '.join([str(ind), str((float(lineParts[5]) + float(lineParts[4])) / 2),
                              str((float(lineParts[7]) + float(lineParts[6])) / 2),
                              str(float(lineParts[5]) - float(lineParts[4])),
                              str(float(lineParts[7]) - float(lineParts[6]))]) + '\n')

print("Annotation Count : " + str(cnt))
# 去重
commands = list(set(commands))
print("Number of images to be downloaded : " + str(len(commands)))
# imap并行计算
list(tqdm(pool.imap(os.system, commands), total=len(commands)))
pool.close()
pool.join()
