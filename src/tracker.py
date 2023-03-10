import cv2.legacy
import sys
import time
from params import PARAMS
from utils2 import *
from Logger import ConsoleLogger
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor

import numpy as np

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
        '''given a frame, use the past references to make a prediction of the frame'''
        pass

    @abstractmethod
    def restart_tracker(self, frame, info):
        '''re-create tracker with new "ground-truth" labels to prevent drifting'''
        pass

    @abstractmethod
    def init_multiprocessing(self):
        '''initializes variables for multiprocessing'''
        pass

    @abstractmethod
    def waiting_step(self, frame):
        '''step executed when client is waiting for the server data'''
        pass

    @abstractmethod
    def execute_catchup(self, old_timestep, old_outputs) -> {}:
        '''step executed when client receives server data'''
        pass

    @abstractmethod
    def reset_mp(self):
        '''step executed after client re-syncs catchup information'''
        pass

class BoxTracker(Tracker):
    '''Tracker uses format xyhw – change bbox inputs and outputs'''
    def check_addframe_constant(self, nframes : int = PARAMS['WAITING_CONSTANT_N']):
        self.waiting_iteration += 1
        if self.waiting_iteration == nframes:
            self.waiting_iteration = 0
            return True
        return False

    def reset_waiting_policy_constant(self):
        self.waiting_iteration = 0

    def _init_waiting_policy(self, waiting_policy):
        if waiting_policy == 'CONSTANT':
            self.logger.log_info(f'Setting up constant waiting policy with addframe at {PARAMS["WAITING_CONSTANT_N"]}')
            self.waiting_iteration = 0
            self.check_addframe = self.check_addframe_constant
            self.reset_waiting_policy = self.reset_waiting_policy_constant
        elif waiting_policy is None:
            self.logger.log_info('Default waiting policy (add every frame for catchup)')
            pass
        else:
            raise NotImplementedError(f'No other waiting policy implemented: {waiting_policy}')

    def _init_catchup_threads(self, flag, num_threads, min_objects):
        self.mp_catchup = flag
        if not flag:
            return

        self.parallel_executor = ThreadPoolExecutor(max_workers=num_threads)
        self.parallel_thread = None
        self.max_threads = num_threads
        self.min_objects = min_objects

    def __init__(self, logger : ConsoleLogger, tracker_type = PARAMS['TRACKER'],
                 waiting_policy = PARAMS['WAITING_POLICY'], mp_catchup = PARAMS['CATCHUP_MULTITHREAD'],
                 mp_threads = PARAMS['NUM_CATCHUP_THREADS'], mp_objs_pthread = PARAMS['MIN_OBJECTS_THREAD']):
        self.tracker_type = tracker_type
        self.trackers = {} # id : tracker
        self.logger = logger
        self.check_addframe = lambda : True # default policy, should return the boolean and update any hidden states
        self.reset_waiting_policy = lambda : None
        self._init_waiting_policy(waiting_policy)
        self._init_catchup_threads(mp_catchup, mp_threads, mp_objs_pthread)

    def get_new_trackers(self, frame : np.ndarray, detections : {int : (int,)}):
        '''Creates trackers for a new set of Detections
        Detections in the format {id : box_xyxy}
        '''
        trackers = {}
        for object_id, box in detections.items():
            trackers[object_id] = init_tracker(self.tracker_type)
            trackers[object_id].init(frame, map_xyxy_to_xyhw(box))

        return trackers

    def restart_tracker(self, frame : np.ndarray, detections : {int : (int)}):
        self.trackers = self.get_new_trackers(frame, detections)
        if self.mp_catchup:
            self.parallel_thread = None # TODO: cancel

        self.reset_waiting_policy()

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

    def process_frame_with_objects(self, frame, objects_to_track : {int}, target_tracker : str = None) -> {int : (int)}:
        '''returns {object_id, new_bb_xyxy}'''
        if target_tracker is None:
            trackers = self.trackers
        elif target_tracker == 'mp':
            trackers = self.mp_trackers
        else:
            raise NotImplementedError

        updated_boxes = {}
        for object_id, tracker in trackers.items():
            if object_id not in objects_to_track:
                continue
            success, bbox_xyhw = tracker.update(frame)

            if success:
                updated_boxes[object_id] = map_xyhw_to_xyxy(bbox_xyhw)

        return updated_boxes

    def init_multiprocessing(self):
        self.catchup_frames = []
        self.mp_trackers = {}

    @abstractmethod
    def waiting_step(self, frame):  # step executed to prepare for the new data
        if self.check_addframe():
            self.catchup_frames.append(frame)

    @abstractmethod
    def reset_mp(self):
        self.catchup_frames = []
        self.mp_trackers = {}
        self.reset_waiting_policy()

    def execute_catchup_with_objects(self, old_timestep, old_detections, objects_to_track : {int}):
        '''executes catchup with select object ids (select trackers)'''
        stats_to_return = {}
        starting_length = len(self.catchup_frames)
        self.logger.log_debug(f'Starting from {old_timestep}, processing {starting_length}')
        num_processed = 0
        # from _init_trackers()
        self.mp_trackers = self.get_new_trackers(self.catchup_frames[0], old_detections)
        curr_detections = old_detections  # do nothing with old detections
        starting_time = time.time()
        while num_processed < len(self.catchup_frames):  # catchup_frames is dynamic
            if num_processed > PARAMS['CATCHUP_LIMIT']:
                raise AssertionError(f'Catchup taking too many iterations: {len(self.catchup_frames)}')

            curr_detections = self.process_frame_with_objects(self.catchup_frames[num_processed], target_tracker='mp',
                                                              objects_to_track=objects_to_track)
            self.logger.log_debug(f'Processed frame {num_processed}')
            num_processed += 1

        return {'process_time': time.time() - starting_time,
                'added_frames': len(self.catchup_frames) - starting_length}

    def execute_catchup_mp(self, old_timestep, old_detections):
        '''for when the catchup should be multithreaded'''
        if len(old_detections) < self.min_objects:
            self.logger.log_info(f'{len(old_detections)} detections less than {self.min_objects}; doing single threaded')
            return self.execute_catchup_with_objects(old_timestep, old_detections, set(self.trackers.keys()))

        self.logger.log_debug('Launching catchup, multithreaded tracker')
        starting_length = len(self.catchup_frames)
        starting_time = time.time()
        self.mp_trackers = self.get_new_trackers(self.catchup_frames[0], old_detections)
        # spawn threads
        # objects to track, change this variable to become an empty list when all are being "tracked"
        objects_to_track = list(old_detections.keys())
        # no need to randomize objects_to_track
        threads = []
        if len(old_detections) > self.max_threads * self.min_objects:
            objects_per_thread = len(old_detections) // self.max_threads
            remaining_objects = objects_per_thread + len(old_detections) % self.max_threads
            self.logger.log_debug(f'Threads are tracking {objects_per_thread} objects')

            # submitting threads
            threads.append(self.parallel_executor.submit(self.execute_catchup_with_objects, old_timestep,
                                                         old_detections, objects_to_track[:remaining_objects]))
            objects_to_track = objects_to_track[remaining_objects:]
            for i in range(self.max_threads - 1):
                threads.append(self.parallel_executor.submit(self.execute_catchup_with_objects, old_timestep,
                                                             old_detections, objects_to_track[:objects_per_thread]))
                objects_to_track = objects_to_track[objects_per_thread]

        else:
            while len(objects_to_track) > self.min_objects:
                threads.append(self.parallel_executor.submit(self.execute_catchup_with_objects, old_timestep,
                                                             old_detections, objects_to_track[:self.min_objects]))
                objects_to_track = objects_to_track[self.min_objects:]
            else:
                if len(objects_to_track) > 0:
                    threads.append(self.parallel_executor.submit(self.execute_catchup_with_objects, old_timestep,
                                                                 old_detections, objects_to_track))
                    objects_to_track = []

        num_threads = len(threads)

        self.logger.log_info(f'Launched {num_threads} threads')
        assert len(objects_to_track) == 0, f'Threads did not completely fill objects_to_track: {objects_to_track}'
        assert len(threads) > 1, f'Threads length is {num_threads}; mistake'

        # wait for threads to finish running
        # threads will modify the trackers in-place
        time_per_loop = 0.05 # TODO: parameterize
        previous_iter = time.time()
        self.logger.log_debug('Starting loop to check thread status')
        while True:
            if time.time() - previous_iter < time_per_loop:
                time.sleep(time_per_loop/4)

            previous_iter = time.time()
            # check on every thread
            for thread in threads:
                if thread.done():
                    if thread.exception():
                        err = thread.exception()
                        raise NotImplementedError(f'Catchup thread errored for some reason: {str(err)}')
                else:
                    break # break out of for loop, back to while loop
            else:
                break # break out of while loop, every thread is done

        return {'process_time': time.time() - starting_time,
                'added_frames': len(self.catchup_frames) - starting_length,
                'num_threads' : num_threads}

    def execute_catchup(self, old_timestep, old_detections):
        '''executes catchup; '''
        if self.mp_catchup:
            return self.execute_catchup_mp(old_timestep, old_detections)
        else:
            return self.execute_catchup_with_objects(old_timestep, old_detections, set(self.trackers.keys()))

class MaskTracker(Tracker):
    def __init__(self, logger : ConsoleLogger, tracker_type = PARAMS['TRACKER'],
                 waiting_policy = PARAMS['WAITING_POLICY']):
        self.current_mask = None # don't really need the current_mask
        self.logger = logger
        self.tracker_type = tracker_type
        self.box_tracker = BoxTracker(logger, waiting_policy=waiting_policy)
        self.object_references = None

    def get_boxes(self, masks : {int : np.array}) -> {int : [int]}:
        return {object_id : get_bbox_from_mask(mask) for object_id, mask in masks.items()}

    def restart_tracker(self, frame, mask : {int : np.array}):
        self.current_mask = mask
        self.object_references = get_kmean_references(mask, frame)
        self.box_tracker.restart_tracker(frame, self.get_boxes(mask))

    def process_frame(self, frame):
        new_boxes = cast_bbox_to_int(self.box_tracker.process_frame(frame))
        self.current_mask = segment_image_kmeans(frame, new_boxes, self.object_references)
        return self.current_mask

    def init_multiprocessing(self):
        self.old_frame = None
        self.box_tracker.init_multiprocessing()

    def waiting_step(self, frame):  # step executed to prepare for the new data
        if self.old_frame is None:
            self.old_frame = frame
        self.box_tracker.waiting_step(frame)

    def execute_catchup(self, old_timestep, old_mask) -> {}:
        # TODO: reformatting of mask required?
        # TODO: fix old_frame technique
        return_stats = self.box_tracker.execute_catchup(old_timestep, self.get_boxes(old_mask))
        # rebuild old reference after executing catchup
        self.object_references = get_kmean_references(old_mask, self.old_frame)
        return return_stats

    def reset_mp(self):
        self.old_frame = None
        self.box_tracker.reset_mp()

class BoxMaskTracker(Tracker):
    '''tracker that takes as input boxes but outputs masks in the process_frame code'''
    def __init__(self, logger : ConsoleLogger, segmenter = PARAMS['BBOX_SEG'], tracker_type = PARAMS['TRACKER'],
                 waiting_policy = PARAMS['WAITING_POLICY']):
        self.segmenter = segmenter
        if segmenter == 'knn': # nearest neighbor clustering
            self.segment_image = segment_image_kmeans
        elif segmenter == 'mrf': # markov random fields
            self.segment_image = segment_image_mrf
        else:
            raise NotImplementedError

        self.logger = logger
        self.box_tracker = BoxTracker(logger, tracker_type, waiting_policy=waiting_policy)
        self.object_references = None
        self.current_processed_masks = None

    def process_frame(self, frame):
        boxes = self.box_tracker.process_frame(frame)

        if self.segmenter == 'knn':
            self.current_processed_masks = segment_image_kmeans(frame, cast_bbox_to_int(boxes), self.object_references)
            return boxes

        raise NotImplementedError(f'Segmenter not implemented : {self.segmenter}')

    def restart_tracker(self, frame, pred_boxes : {int : np.array}):
        # build references
        if self.segmenter == 'knn':
            self.object_references = get_midbox_references(cast_bbox_to_int(pred_boxes), frame)
        self.box_tracker.restart_tracker(frame, pred_boxes)
        self.current_processed_masks = segment_image_kmeans(frame, cast_bbox_to_int(pred_boxes), self.object_references)

    def init_multiprocessing(self):
        self.old_frame = None
        self.box_tracker.init_multiprocessing()

    def waiting_step(self, frame):  # step executed to prepare for the new data
        if self.old_frame is None:
            self.old_frame = frame
        self.box_tracker.waiting_step(frame)

    def execute_catchup(self, old_timestep, old_outputs) -> {}:
        return_info = self.box_tracker.execute_catchup(old_timestep, old_outputs)
        if self.segmenter == 'knn':
            self.object_references = get_midbox_references(cast_bbox_to_int(old_outputs), self.old_frame)
        return return_info

    def get_masks(self):
        return self.current_processed_masks

    def reset_mp(self):
        self.old_frame = None
        self.box_tracker.reset_mp()