import json
import cv2
import numpy as np

import torch
from datetime import datetime
import os
import os.path as osp

import mmcv

from mmpose.mmpose.apis import (inference_bottom_up_pose_model, init_pose_model, vis_pose_result)
from mmpose.mmpose.datasets import DatasetInfo

from mmpose.skeleton_utils import *


yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', 'mmpose/yolo_models/model_4.pt', force_reload=True)
pose_model = init_pose_model('mmpose/associative_embedding_hrnet_w32_coco_512x512.py', 'mmpose/hrnet_w32_coco_512x512-bcb8c247_20200816.pth', device='cuda:0')
dataset = pose_model.cfg.data['test']['type']
dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
dataset_info = DatasetInfo(dataset_info)
video_file = 'videos/1.mp4'

start_point, end_point, class_name = get_zones(video_file, yolo_model)

text_flag = False
flag = False
time = datetime.now()
text_time = datetime.now()
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap = cv2.VideoCapture(video_file)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('outputs/fall_1.mp4', fourcc, 30.0, (640, 480))
i = 0
tmp = 0
falls = 0
object_time = 0
color = (255, 0, 0)
thickness = 2

while cap.isOpened():
    i += 1
    ret, frame = cap.read()

    frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
    frame = cv2.putText(frame, class_name, (end_point[0]-50, start_point[1]-7), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA, False)
    if isinstance(frame, np.ndarray):
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
    else: break

    # frame = cv2.rotate(frame, cv2.ROTATE_180)
    if i % 60 == 0 or i == 1:
        pose_results, returned_outputs = inference_bottom_up_pose_model(
                    pose_model,
                    frame,
                    dataset=dataset,
                    dataset_info=dataset_info,
                    pose_nms_thr=0.9,
                    return_heatmap=False,
                    outputs=None)
        
        if len(pose_results) > 0:
            if pose_results[0]['score'] > 0.05:
                points = pose_results[0]['keypoints'].tolist()
                points = [[int(x[0]), int(x[1])] for x in points]
                point_0 = (points[0][0],points[0][1])
                point_2 = (points[15][0] + int((points[16][0] - points[15][0])/2), points[15][1] + int((points[16][1] - points[15][1])/2))
                vector = make_vector(point_0, point_2)
                length = get_len(vector)

                points_pose = sum([points_positions(point, start_point, end_point) for point in points])

                if points_pose >= 7:
                    object_time += 1

                if i - tmp > 140:
                    text_flag = False
                    
                if flag:
                    angle = get_angle(last_vector, vector)
                    # print(angle)
                    if angle > 1.1 and angle < 1.57 and vector[1] < last_vector[1]:
                        # print(length, last_length)
                        text_flag = True
                        tmp = i
                        text_time = datetime.now()
                        falls += 1
                flag = True
                last_vector = vector
                last_length = length
                time = datetime.now()

    if text_flag:
        frame = cv2.putText(frame, 'Fall detected', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA, False)

    if flag:
        frame = draw_skeleton(frame, points)
        frame = cv2.line(frame, point_0, point_2, [0, 0, 255], 2)
    out.write(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


frame = np.zeros((480, 640, 3), np.uint8)
frame = cv2.putText(frame, f'Falls: {falls}', (120, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA, False)
frame = cv2.putText(frame, f'Time in {class_name}: {object_time}', (120, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA, False)

for _ in range(50):
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
