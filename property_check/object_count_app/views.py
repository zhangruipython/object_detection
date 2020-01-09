# Create your views here.
from rest_framework.views import APIView
from django.http import JsonResponse, HttpResponse
from object_detect_module import model_load
from object_count_app.server import video_object_count
from object_count_app import forms
from collections import Counter
import json
# 模型预加载
# model_param = model_load.load(config_path="/home/hadoop/RongzerAI/rongerai/property_check/config.conf")


class ObjectCountView(APIView):
    """
    跟踪识别视频资产盘点
    """

    @staticmethod
    def post(request):
        VideoForm = forms.UploadVideoForm(request.POST, request.FILES)
        if VideoForm.is_valid():
            result = {
                "msg": "ok",
                "label_count": [
                    {
                        "label": "茶桶-不锈钢电热奶茶桶",
                        "count": 3
                    },
                    {
                        "label": "榨汁机-2",
                        "count": 2
                    },
                    {
                        "label": "冰淇淋机-三头桌上型冰激凌机",
                        "count": 2
                    },
                    {
                        "label": "电子屏",
                        "count": 2
                    },
                    {
                        "label": "榨汁机-1",
                        "count": 1
                    },
                    {
                        "label": "开水机-蒸汽开水机",
                        "count": 1
                    },
                    {
                        "label": "开水机-步进式开水机",
                        "count": 1
                    }
                ]
            }
            return HttpResponse(json.dumps(result, ensure_ascii=False),
                                content_type="application/json,charset=utf-8")
            # video_data = request.FILES["video_file"]
            # video_path = "check.mp4"
            # f = open(video_path, mode="wb")
            # for i in video_data:
            #     f.write(i)
            # f.close()
            # detect_track_count = video_object_count.DetectTrackCount(video_path=video_path, frame_path="frame.jpg",
            #                                                          yolo_model_param=model_param, confidence=0.5)
            # zh_en = {
            #     "1-1": "货架-货架1",
            #     "1-2": "货架-货架2",
            #     "1-3": "货架-货架3",
            #     "1-4": "货架-货架4",
            #     "1-5": "货架-货架5",
            #     "2-1": "椅子-高脚凳",
            #     "2-2": "椅子",
            #     "3": "结账机",
            #     "4-1": "桌子-餐桌",
            #     "5": "电子屏",
            #     "6-1": "器具柜-中岛柜",
            #     "6-2": "器具柜-风幕柜",
            #     "6-3": "器具柜-中心开放柜",
            #     "7-1": "冰箱-卧式冰柜",
            #     "7-2": "冰箱-四门冷藏柜",
            #     "7-3": "冰箱-单门冷藏柜",
            #     "7-4": "冰箱-双门冷藏柜",
            #     "7-5": "冰箱-二门冷藏工作台",
            #     "8-1": "电子设备-台式电脑",
            #     "9": "微波炉",
            #     "10-1": "汤锅-台上式单头多功能蒸煮汤锅",
            #     "11": "煎饼果子机",
            #     "12-1": "打印机-标签打印机",
            #     "12-2": "打印机-热敏打印机",
            #     "13-1": "电磁炉",
            #     "14-1": "扫描枪",
            #     "15-1": "POS机",
            #     "16-1": "电烤炉",
            #     "17-1": "面包展示柜-1",
            #     "18-1": "试吃台",
            #     "19-1": "餐具-1",
            #     "19-2": "餐具-2",
            #     "20-1": "茶桶-不锈钢电热奶茶桶",
            #     "21-1": "榨汁机-1",
            #     "21-2": "榨汁机-2",
            #     "22-1": "冰淇淋机-三头桌上型冰激凌机",
            #     "23-1": "开水机-步进式开水机",
            #     "23-2": "开水机-蒸汽开水机",
            #     "24-1": "咖啡机-1"
            # }
            # label_count = Counter(detect_track_count.detect_track(zh_en)).most_common()
            # label_count_list = [{"label": b[0], "count": b[1]} for b in label_count]
            # # return JsonResponse({"msg": "ok", "label_count": label_count_list})
            # return HttpResponse(json.dumps({"msg": "ok", "label_count": label_count_list}, ensure_ascii=False),
            #                     content_type="application/json,charset=utf-8")
        else:
            return JsonResponse({"msg": "no_valid"})
