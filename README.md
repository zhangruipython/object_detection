# object_detection
在yolov3算法的基础上进行目标检测应用开发
## 主要功能模块有：
### CPU版本目标识别    ------> 通过调用OpenCV dnn模块实现yolov3算法的目标识别，由于OpenCV对cpu多核多进程的优化识别速度比原生的darknet cpu版本快
### GPU版本目标识别    ------> 通过GPU cuda加速实现高速识别，是目前所能实现的速度最快的版本
### 大规模图像训练数据集下载    ------> openimage v4 数据集下载 
### 视频中目标数量统计 ------> deepsort多目标跟踪算法结合yolov3目标检测算法的视频目标统计
