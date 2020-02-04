import os
import sys
import cv2
import csv
import time
import json
import torch
import PIL.Image
import numpy as np
import utils
from operator import itemgetter
from sklearn.utils.linear_assignment_ import linear_assignment

from pprint import pprint

import trt_pose.coco
import trt_pose.models
from torch2trt import TRTModule
import torchvision.transforms as transforms
from pose import *




def IOU(boxA, boxB):
    # pyimagesearch: determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def get_bbox(kp_list):
    bbox = []
    for aggs in [min, max]:
        for idx in range(2):
            bound = aggs(kp_list, key=itemgetter(idx))[idx]
            bbox.append(bound)
    return bbox

def tracker_match(trackers, detections, iou_thrd = 0.3):
    '''
    From current list of trackers and new detections, output matched detections,
    unmatched trackers, unmatched detections.
    https://towardsdatascience.com/computer-vision-for-tracking-8220759eee85
    '''

    IOU_mat= np.zeros((len(trackers),len(detections)),dtype=np.float32)
    for t,trk in enumerate(trackers):
        for d,det in enumerate(detections):
            IOU_mat[t,d] = IOU(trk,det)

    # Produces matches
    # Solve the maximizing the sum of IOU assignment problem using the
    # Hungarian algorithm (also known as Munkres algorithm)

    matched_idx = linear_assignment(-IOU_mat)

    unmatched_trackers, unmatched_detections = [], []
    for t,trk in enumerate(trackers):
        if(t not in matched_idx[:,0]):
            unmatched_trackers.append(t)

    for d, det in enumerate(detections):
        if(d not in matched_idx[:,1]):
            unmatched_detections.append(d)

    matches = []

    # For creating trackers we consider any detection with an
    # overlap less than iou_thrd to signifiy the existence of
    # an untracked object

    for m in matched_idx:
        if(IOU_mat[m[0],m[1]] < iou_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1,2))

    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

source = sys.argv[1]
source = int(source) if source.isdigit() else source
cap = cv2.VideoCapture(source)

w = int(cap.get(3))
h = int(cap.get(4))

fourcc_cap = cv2.VideoWriter_fourcc(*'MJPG')
cap.set(cv2.CAP_PROP_FOURCC, fourcc_cap)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

DEBUG = True
WRITE2CSV = False
WRITE2VIDEO = False
RUNSECONDARY = True

if WRITE2CSV:
    activity = os.path.basename(source)
    dataFile = open('data/{}.csv'.format(activity),'w')
    newFileWriter = csv.writer(dataFile)

if WRITE2VIDEO:
    # Define the codec and create VideoWriter object
    name = 'out.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(name, fourcc, 30.0, (w, h))

if RUNSECONDARY:
    import tensorflow as tf
    secondary_model = tf.keras.models.load_model('models/lstm.h5')
    window = 5
    pose_vec_dim = 36
    motion_dict = {0: 'squat', 1: 'deadlift', 2: 'stand'}


trackers = []
while True:

    ret, frame = cap.read()
    bboxes = []
    if ret:

        image, pose_list = inference(frame)
        for body in pose_list:
            if body:
                bbox = get_bbox(list(body.values()))
                bboxes.append((bbox, body))

        trackers = utils.update_trackers(trackers, bboxes)

        if RUNSECONDARY:
            for tracker in trackers:
                print(len(tracker.q))
                if len(tracker.q) >= window:
                    sample = np.array(list(tracker.q)[:window])
                    sample = sample.reshape(1, pose_vec_dim, window)
                    pred_activity = motion_dict[np.argmax(secondary_model.predict(sample)[0])]
                    tracker.activity = pred_activity
                    image = tracker.annotate(image)
                    print(pred_activity)

        if DEBUG:
            pprint([(tracker.id, np.vstack(tracker.q)) for tracker in trackers])

        if WRITE2CSV:
            for tracker in trackers:
                print(len(tracker.q))
                if len(tracker.q) >= 3:
                    newFileWriter.writerow([activity] + list(np.hstack(list(tracker.q)[:3])))

        if WRITE2VIDEO:
            out.write(image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()

try:
    dataFile.close()
except:
    pass

try:
    out.release()
except:
    pass
