import cv2.legacy
import sys
import time
from params import PARAMS
import numpy as np
from utils2 import *
from Logger import ConsoleLogger
from abc import abstractmethod

import numpy as np
from scipy.optimize import linear_sum_assignment

def segment_subimg_mrf(subimg):
    height, width = subimg.shape[0], subimg.shape[1]

    # Define the MRF parameters
    n_states = 2
    smoothness = 1.0

    smooth_energy = 0
    for i in range(height):
        for j in range(width):
            if i < height - 1:
                smooth_energy += smoothness * np.sum(np.abs(subimg[i, j] - subimg[i + 1, j]))
            if j < width - 1:
                smooth_energy += smoothness * np.sum(np.abs(subimg[i, j] - subimg[i, j + 1]))

    # Optimize the MRF using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(np.array([[smooth_energy] * n_states]))

    # Get the optimized binary segmentation mask
    mask = np.zeros((height, width), dtype=np.int8)
    mask[row_ind[0]] = col_ind

    return mask

def get_subimg_midbox(subimg, box_radius=5):
    '''get the midbox of a subimg (the bounding box) for use as reference cluster'''
    midpoint = subimg.shape // 2
    midbox = subimg[midpoint[0] - box_radius: midpoint[0] + box_radius,
             midpoint[1] - box_radius: midpoint[1] + box_radius]

    return midbox

def segment_subimg_knn(subimg, bg_ref, norm=1, box_radius=5) -> np.array:
    '''segments a bounding box with a given subimg and a background reference'''
    midbox = get_subimg_midbox(subimg, box_radius)

    midbox_avg = midbox.mean(axis=(0, 1))

    bg_dists = (subimg - bg_ref) ** 2 / norm
    ref_dists = (subimg - midbox_avg) ** 2 / norm

    return ref_dists.sum(axis=-1) > bg_dists.sum(axis=-1)

def segment_image_knn(full_img: np.array, object_boxes: {int: [int]}, box_radius=5):
    '''for every box mapped as object_id : [bbox (XYWH)], return the cluster
    full_img should be H x W x 3'''
    assert full_img.shape[2] == 3

    img_lab = rgb2lab(full_img)
    image_normalization = img_lab.max(axis=(0, 1))

    zero_img = np.zeros_like(img_lab, dtype=bool)
    zero_img[:5] = 1
    zero_img[-5:] = 1
    zero_img[:, -5:] = 1
    zero_img[:, :5] = 1

    ref_bg = img_lab[zero_img].reshape(-1, 3).mean(axis=0)

    obj_masks = {}

    for object_id, bbox in object_boxes.items():
        if bbox[2] < box_radius or bbox[3] < box_radius:
            continue

        subimg_lab = img_lab[bbox[0]: bbox[0] + bbox[2], bbox[1]: bbox[1] + bbox[3]]
        obj_masks[object_id] = segment_subimg_knn(subimg_lab, ref_bg, image_normalization, box_radius)

    return obj_masks

def segment_image_mrf(full_img : np.array, object_boxes : {int : [int]}):
    obj_masks = {}
    for object_id, bbox in object_boxes.items():
        subimg = full_img[bbox[0] : bbox[0] + bbox[2], bbox[1] : bbox[1] + bbox[3]].copy()
        mask = segment_subimg_mrf(subimg)

        obj_masks[object_id] = mask

    return obj_masks


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

class Tracker:
    @abstractmethod
    def process_frame(self, frame):
        pass

    @abstractmethod
    def restart_tracker(self, frame, info):
        pass

    @abstractmethod
    def init_multiprocessing(self):
        pass

    @abstractmethod
    def waiting_step(self, frame):  # step executed to prepare for the new data
        pass

    @abstractmethod
    def execute_catchup(self, old_timestep, old_outputs) -> {}:
        pass

    @abstractmethod
    def reset_mp(self):
        pass

class BoxTracker(Tracker):
    '''Tracker uses format xyhw – change bbox inputs and outputs'''
    def __init__(self, logger : ConsoleLogger, tracker_type = PARAMS['TRACKER']):
        self.tracker_type = tracker_type
        self.trackers = {} # id : tracker
        self.logger = logger

    def restart_tracker(self, frame : np.ndarray, detections : {int : (int)}, target_tracker : str = None):
        '''Creates trackers for a new set of Detections
        Detections in the format {class : {id : box_xyxy}}
        '''
        trackers = {}
        for object_id, box in detections.items():
            trackers[object_id] = init_tracker(self.tracker_type)
            trackers[object_id].init(frame, map_xyxy_to_xyhw(box))

        if target_tracker is None:
            self.trackers = trackers
        elif target_tracker == 'mp':
            self.mp_trackers = trackers
        else:
            raise NotImplementedError

    def process_frame(self, frame, target_tracker : str = None) -> {int : (int)}:
        '''returns {object_id, new_bb_xyxy}'''

        if target_tracker is None:
            trackers = self.trackers
        elif target_tracker == 'mp':
            trackers = self.mp_trackers
        else:
            raise NotImplementedError

        updated_boxes = {}
        for object_id, tracker in trackers.items():
            success, bbox_xyhw = tracker.update(frame)

            if success:
                updated_boxes[object_id] = map_xyhw_to_xyxy(bbox_xyhw)

        return updated_boxes

    def init_multiprocessing(self):
        self.catchup_frames = []
        self.mp_trackers = {}

    @abstractmethod
    def waiting_step(self, frame):  # step executed to prepare for the new data
        self.catchup_frames.append(frame)

    @abstractmethod
    def reset_mp(self):
        self.catchup_frames = []
        self.mp_trackers = {}

    def execute_catchup(self, old_timestep, old_detections):
        stats_to_return = {}
        starting_length = len(self.catchup_frames)
        self.logger.log_debug(f'Starting from {old_timestep}, processing {starting_length}')
        num_processed = 0
        # from _init_trackers()
        self.restart_tracker(self.catchup_frames[0], old_detections, target_tracker='mp')
        curr_detections = old_detections  # do nothing with old detections
        starting_time = time.time()
        while num_processed < len(self.catchup_frames):  # catchup_frames is dynamic
            if num_processed > 10:
                raise AssertionError('Catchup taking too many iterations')

            curr_detections = self.process_frame(self.catchup_frames[num_processed], target_tracker='mp')
            self.logger.log_debug(f'Processed frame {num_processed}')
            num_processed += 1

        return {'process_time' : time.time() - starting_time,
                'added_frames' : len(self.catchup_frames) - starting_length}

class MaskTracker(Tracker):
    def __init__(self, logger : ConsoleLogger, tracker_type = PARAMS['TRACKER']):
        self.current_mask = None
        self.logger = logger
        self.tracker_type = tracker_type

    def restart_tracker(self, frame, mask):
        self.current_mask = mask

    def process_frame(self, frame):
        return self.current_mask

    def init_multiprocessing(self):
        pass

    def waiting_step(self, frame):  # step executed to prepare for the new data
        pass

    def execute_catchup(self, old_timestep, old_mask) -> {}:
        # TODO: reformatting of mask required?
        return old_mask

    def reset_mp(self):
        pass

class BoxMaskTracker(Tracker):
    def __init__(self, logger : ConsoleLogger, segmenter = PARAMS['BBOX_SEG'], tracker_type = PARAMS['TRACKER']):
        if segmenter == 'knn': # nearest neighbor clustering
            self.segment_image = segment_image_knn
        elif segmenter == 'mrf': # markov random fields
            self.segment_image = segment_image_mrf
        else:
            raise NotImplementedError

        self.logger = logger
        self.tracker = BoxTracker(tracker_type)

    def process_frame(self, frame):
        objects = self.tracker.process_frame(frame)

        return self.segment_image(frame, self.round_tracker_outputs(objects))

    def get_pred_boxes_from_masks(self, pred_masks : {int : np.array}) -> {int : (int,)}:
        pred_boxes = {}
        for object_id, box in pred_masks.items():
            pred_boxes[object_id] = get_bbox_from_mask(box)

        return pred_boxes

    def round_tracker_outputs(self, objects):
        return {k : v.astype(int) for k, v in objects.items()}

    def restart_tracker(self, frame, pred_masks : {int : np.array}):
        self.tracker.restart_tracker(frame, self.get_pred_boxes_from_masks(pred_masks))

    def init_multiprocessing(self):
        self.tracker.init_multiprocessing()

    def waiting_step(self, frame):  # step executed to prepare for the new data
        self.tracker.waiting_step(frame)

    def execute_catchup(self, old_timestep, old_outputs) -> {}:
        return self.tracker.execute_catchup(old_timestep, self.get_pred_boxes_from_masks(old_outputs))

    def reset_mp(self):
        self.tracker.reset_mp()