# -*- coding: utf-8 -*-
"""
@Time    : 2019-12-09 16:26
@Author  : zhangrui
@FileName: demo.py
@Software: PyCharm
"""
import cv2
from random import randint

trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']


def createTrackerByName(trackerType):
    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)

    return tracker


if __name__ == '__main__':
    trackerType = "MEDIANFLOW"

    # Set video to load
    videoPath = "C:\\rongze\\data\\yike_picture\\lukuang.mp4"

    # Create a video capture object to read videos
    cap = cv2.VideoCapture(videoPath)

    # Read first frame
    success, frame = cap.read()
    # quit if unable to read the video file
    if not success:
        print('Failed to read video')

    bboxes = [(572, 279, 235, 201), (427, 392, 66, 58)]
    colors = [(randint(64, 255), randint(64, 255), randint(64, 255)),
              (randint(64, 255), randint(64, 255), randint(64, 255))]
    box = (246, 3, 1204, 593)
    color = (randint(64, 255), randint(64, 255), randint(64, 255))
    # (572, 279, 235, 201) (427, 392, 66, 58)
    # for i in range(2):
    #     bbox = cv2.selectROI('MultiTracker', frame)
    #     bboxes.append(bbox)
    #     colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))

    print('Selected bounding boxes {}'.format(bboxes))

    # multiTracker = cv2.MultiTracker_create()
    tracker = cv2.TrackerKCF_create()
    # Initialize MultiTracker
    # for bbox in bboxes:
    #     multiTracker.add(createTrackerByName(trackerType), frame, bbox)
    tracker.init(frame, box)

    # Process video and track objects
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # get updated location of objects in subsequent frames
        success, boxes = tracker.update(frame)
        print(success, boxes)
        cv2.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[0] + boxes[2]), int(boxes[1] + boxes[3])),colors[1],2,1)
        # draw tracked objects
        # for i, newbox in enumerate(boxes):
        #     print(newbox)
        #     p1 = (int(newbox[0]), int(newbox[1]))
        #     p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        #     cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

        # show frame
        cv2.imshow('MultiTracker', frame)

        # quit on ESC button
        if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
            break
