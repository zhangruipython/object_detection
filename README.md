# object_detection
在yolov3算法的基础上进行目标识别的开发
## 主要功能模块有：
### CPU版本目标识别    ------> 通过调用OpenCV dnn模块实现yolov3算法的目标识别，由于OpenCV对cpu多核多进程的优化识别速度比原生的darknet cpu版本快
### GPU版本目标识别    ------> 通过GPU cuda加速实现高速识别，是目前所能实现的速度最快的版本
### openimage v4 数据集下载     ------> https://storage.googleapis.com/openimages/web/index.html
## 实际问题解决方案有：
### 实时视频流计算车辆长度、车辆轮胎数量、车辆前轮到后轮轴距
