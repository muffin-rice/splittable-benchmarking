import cv2.legacy
import sys
import time
from params import PARAMS
import numpy as np
from utils2 import *
from Logger import ConsoleLogger

def init_tracker(tracker_type="MEDIANFLOW"):
    if tracker_type == 'BOOSTING':
        tracker = cv2.legacy.TrackerBoosting_create()
    elif tracker_type == 'MIL':
        tracker = cv2()
    elif tracker_type == 'KCF':
        tracker = cv2.legacy.TrackerKCF_create()
    elif tracker_type == 'TLD':
        tracker = cv2.legacy.TrackerTLD_create()
    elif tracker_type == 'MEDIANFLOW':
        tracker = cv2.legacy.TrackerMedianFlow_create()
    elif tracker_type == 'GOTURN':
        tracker = cv2.legacy.TrackerGOTURN_create()
    elif tracker_type == 'MOSSE':
        tracker = cv2.legacy.TrackerMOSSE_create()
    elif tracker_type == "CSRT":
        tracker = cv2.legacy.TrackerCSRT_create()
    else:
        raise NotImplementedError('No Tracker Found.')

    return tracker


class Tracker():
    '''Tracker uses format xyhw – change bbox inputs and outputs'''
    def __init__(self, tracker_type = PARAMS['TRACKER']):
        self.tracker_type = tracker_type
        self.trackers = {} # id : tracker

    def handle_new_detection(self, frame : np.ndarray, detections : {int : (int)}):
        '''Creates trackers for a new set of Detections
        Detections in the format {class : {id : box_xyxy}}
        '''
        self.trackers = {}
        for object_id, box in detections.items():
            self.trackers[object_id] = init_tracker(self.tracker_type)
            self.trackers[object_id].init(frame, map_xyxy_to_xyhw(box))

    def add_bounding_box(self, frame, bbox_xyxy, object_id):
        raise NotImplementedError
        # self.trackers[object_id] = init_tracker(self.tracker_type)
        # self.trackers[object_id].init(frame, map_xyxy_to_xyhw(bbox_xyxy))

    def update(self, frame) -> {int : (int)}:
        '''returns {object_id, new_bb_xyxy}'''

        updated_boxes = {}
        for object_id, tracker in self.trackers.items():
            success, bbox_xyhw = tracker.update(frame)

            if success:
                updated_boxes[object_id] = map_xyhw_to_xyxy(bbox_xyhw)

        return updated_boxes

