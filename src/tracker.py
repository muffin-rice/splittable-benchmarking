import cv2.legacy
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
        '''step executed after client re-syncs catchup information; should be in-thread'''
        pass

    @abstractmethod
    def close_mp(self):
        '''step executed upon start of a new video; kill all parallel threads, etc.'''
        pass

class BoxTracker(Tracker):
    '''Tracker uses format xyhw – change bbox inputs and outputs'''
    def check_addframe_constant(self, nframes : int = PARAMS['WAITING_CONSTANT_N']):
        self.waiting_iteration += 1
        if self.waiting_iteration % nframes == 0:
            self.waiting_iteration = 0
            return True
        return False

    def reset_waiting_policy_constant(self):
        self.waiting_iteration = -1

    def _init_waiting_policy(self, waiting_policy):
        if waiting_policy == 'CONSTANT':
            self.logger.log_info(f'Setting up constant waiting policy with addframe at {PARAMS["WAITING_CONSTANT_N"]}')
            self.waiting_iteration = -1
            self.check_addframe = self.check_addframe_constant
            self.reset_waiting_policy = self.reset_waiting_policy_constant
        elif waiting_policy is None:
            self.logger.log_info('Default waiting policy (add every frame for catchup)')
            pass
        else:
            raise NotImplementedError(f'No other waiting policy implemented: {waiting_policy}')

    def _init_catchup_threads(self, flag, num_threads, min_objects):
        self.mp_catchup = flag
        if not self.mp_catchup:
            return

        self.parallel_catchup_executor = ThreadPoolExecutor(max_workers=num_threads)
        self.parallel_catchup_threads = [None]
        self.max_threads = num_threads
        self.min_objects = min_objects

    def __init__(self, logger : ConsoleLogger, tracker_type = PARAMS['TRACKER'],
                 waiting_policy = PARAMS['WAITING_POLICY'], mp_catchup = PARAMS['CATCHUP_MULTITHREAD'],
                 mp_threads = PARAMS['NUM_CATCHUP_THREADS'], mp_objs_pthread = PARAMS['MIN_OBJECTS_THREAD'],
                 device=None):
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
            for thread in self.parallel_catchup_threads:
                if thread is not None:
                    thread.cancel()
        self.reset_waiting_policy()

    def process_frame(self, frame) -> ({int : (int,)}, {str : float}):
        '''returns {object_id, new_bb_xyxy}'''
        # TODO: introduce variables if necessary, mp_catchup is separated (cannot share same ThreadPoolExec)
        if self.mp_catchup and len(self.trackers) >= 2*self.min_objects:
            # multithread the catchup
            pass
        else:
            return process_objects_in_tracker(self.trackers, frame, list(self.trackers.keys())), {}

        updated_boxes = {}

        for object_id, tracker in self.trackers.items():
            success, bbox_xyhw = tracker.update(frame)

            if success:
                updated_boxes[object_id] = map_xyhw_to_xyxy(bbox_xyhw)

        return updated_boxes, {}

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
        self.trackers = self.mp_trackers
        self.mp_trackers = {}
        self.reset_waiting_policy()

    @abstractmethod
    def close_mp(self):
        self.catchup_frames = []
        self.mp_trackers = {}
        self.reset_waiting_policy()
        if self.mp_catchup:
            for thread in self.parallel_catchup_threads:
                if thread is not None:
                    thread.cancel()

    def execute_catchup_with_objects(self, old_timestep, old_detections, objects_to_track : {int}):
        '''executes catchup with select object ids (select trackers); singlethreaded only'''
        stats_to_return = {}
        starting_length = len(self.catchup_frames)
        self.logger.log_debug(f'Starting from {old_timestep}, processing {starting_length}')
        num_processed = 0

        curr_detections = old_detections  # do nothing with old detections
        starting_time = time.time()
        while num_processed < len(self.catchup_frames):  # catchup_frames is dynamic
            if num_processed > PARAMS['CATCHUP_LIMIT']:
                raise AssertionError(f'Catchup taking too many iterations: {len(self.catchup_frames)}')

            curr_detections = process_objects_in_tracker(self.mp_trackers, self.catchup_frames[num_processed],
                                                         objects_to_track=objects_to_track)
            self.logger.log_debug(f'Processed frame {num_processed}')
            num_processed += 1

        return {'catchup_time': time.time() - starting_time,
                'added_frames': len(self.catchup_frames) - starting_length}

    def execute_catchup_mp(self, old_timestep, old_detections):
        '''for when the catchup should be multithreaded
        launches many self.execute_catchup_with_objects'''
        if len(old_detections) <= self.min_objects:
            self.logger.log_info(f'{len(old_detections)} detections less than {self.min_objects}; doing single threaded')
            return self.execute_catchup_with_objects(old_timestep, old_detections, set(self.trackers.keys()))

        self.logger.log_debug('Launching catchup, multithreaded tracker')
        starting_length = len(self.catchup_frames)
        starting_time = time.time()
        # spawn threads
        # objects to track, change this variable to become an empty list when all are being "tracked"
        objects_to_track = list(old_detections.keys())
        # no need to randomize objects_to_track
        threads = []

        threads_with_objects = partition_objects_into_threads(objects_to_track, self.max_threads, self.min_objects)

        for thread_objects in threads_with_objects:
            threads.append(self.parallel_catchup_executor.submit(self.execute_catchup_with_objects, old_timestep,
                                                                 old_detections, thread_objects))

        self.parallel_catchup_threads = threads

        num_threads = len(threads)

        self.logger.log_info(f'Launched {num_threads} threads')

        assert len(threads) > 1, f'Threads length is {num_threads}; mistake'

        # wait for threads to finish running
        # threads will modify the trackers in-place
        self.logger.log_debug('Starting loop to check thread status')
        wait_for_threads(threads)

        self.parallel_catchup_threads = [None]

        return {'catchup_time': time.time() - starting_time,
                'added_frames': len(self.catchup_frames) - starting_length,
                'num_threads' : num_threads}

    def execute_catchup(self, old_timestep, old_detections):
        '''executes catchup; '''
        self.mp_trackers = self.get_new_trackers(self.catchup_frames[0], old_detections)
        if self.mp_catchup:
            return self.execute_catchup_mp(old_timestep, old_detections)
        else:
            return self.execute_catchup_with_objects(old_timestep, old_detections, set(self.trackers.keys()))

class MaskTracker(Tracker):
    def __init__(self, logger : ConsoleLogger, tracker_type = PARAMS['TRACKER'],
                 waiting_policy = PARAMS['WAITING_POLICY'], device = PARAMS['COMPRESSOR_DEVICE']):
        self.current_mask = None # don't really need the current_mask
        self.logger = logger
        self.tracker_type = tracker_type
        self.box_tracker = BoxTracker(logger, waiting_policy=waiting_policy)
        self.object_references = None
        self.device = device

    def get_boxes(self, masks : {int : np.array}) -> {int : [int]}:
        return {object_id : get_bbox_from_mask(mask) for object_id, mask in masks.items()}

    def restart_tracker(self, frame, mask : {int : np.array}):
        self.current_mask = mask
        self.object_references = get_kmean_references(mask, frame)
        self.box_tracker.restart_tracker(frame, self.get_boxes(mask))

    def process_frame(self, frame):
        time_dict = {}
        box_time = time.time()
        new_boxes, _ = self.box_tracker.process_frame(frame)
        time_dict['box_time'] = time.time() - box_time
        seg_time = time.time()
        self.current_mask = segment_image_kmeans(frame, cast_bbox_to_int(new_boxes), self.object_references, device = self.device)
        time_dict['seg_time'] = time.time() - seg_time
        return self.current_mask, time_dict

    def init_multiprocessing(self):
        self.old_frame = None
        self.old_object_references = None
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

    def close_mp(self):
        self.old_frame = None
        self.box_tracker.close_mp()

class BoxMaskTracker(Tracker):
    '''tracker that takes as input boxes but outputs masks in the process_frame code'''
    def __init__(self, logger : ConsoleLogger, segmenter = PARAMS['BBOX_SEG'], tracker_type = PARAMS['TRACKER'],
                 waiting_policy = PARAMS['WAITING_POLICY'], device = PARAMS['COMPRESSOR_DEVICE']):
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
        self.device = device

    def process_frame(self, frame):
        '''processes the frame in certain device and then returns in cpu device'''
        time_dict = {}
        box_time = time.time()
        boxes, _ = self.box_tracker.process_frame(frame)
        time_dict['box_time'] = time.time() - box_time

        if self.segmenter == 'knn':
            kmeans_time = time.time()
            self.current_processed_masks = segment_image_kmeans(frame, cast_bbox_to_int(boxes), self.object_references,
                                                                device = self.device)
            time_dict['seg_time'] = time.time() - kmeans_time
            return boxes, time_dict

        raise NotImplementedError(f'Segmenter not implemented : {self.segmenter}')

    def restart_tracker(self, frame, pred_boxes : {int : np.array}):
        # build references
        if self.segmenter == 'knn':
            self.object_references = get_midbox_references(cast_bbox_to_int(pred_boxes), frame)
        self.box_tracker.restart_tracker(frame, pred_boxes)
        self.current_processed_masks = segment_image_kmeans(frame, cast_bbox_to_int(pred_boxes), self.object_references,
                                                            device = self.device)

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
            self.old_object_references = get_midbox_references(cast_bbox_to_int(old_outputs), self.old_frame)

        self.logger.log_debug(f'Reassigned object_references: {self.object_references.keys()}')
        return return_info

    def get_masks(self):
        return self.current_processed_masks

    def reset_mp(self):
        self.old_frame = None
        self.box_tracker.reset_mp()
        self.object_references = self.old_object_references
        self.old_object_references = None

    def close_mp(self):
        self.old_frame = None
        self.old_object_references = None
        self.box_tracker.close_mp()