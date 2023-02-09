from socket import *
from struct import pack, unpack
import torch
from torch import nn

import pickle
import time
import sys
import argparse
from params import PARAMS, CURR_DATE, DESIRED_CLASSES
from Logger import ConsoleLogger, DictionaryStatsLogger
from utils2 import *
from data import Dataset
from tracker import Tracker, BoxTracker, MaskTracker, BoxMaskTracker
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

def create_input(data):
    return data

def get_gt_detections(class_info : ((int,), (int,)), boxes : [[int,],]) -> {int : {int : (int,)}}:
    '''reformats the annotations into eval-friendly output
    class_info is (class_ids, object_ids)
    returns as {class_ids : {object_ids : box}}'''
    assert len(class_info[0]) == len(boxes), f'class_info: {class_info} boxes: {boxes}'
    gt_boxes = {}
    for i in range(len(class_info[0])):
        if class_info[0][i] not in DESIRED_CLASSES:
            continue

        if class_info[0][i] in gt_boxes:
            gt_boxes[class_info[0][i]][class_info[1][i]] = boxes[i]
        else:
            gt_boxes[class_info[0][i]] = {class_info[1][i]: boxes[i]}

    return gt_boxes

def get_gt_detections_as_pred(class_info : ((int,), (int,)), boxes : [[int,],]) -> {int : (int,)}:
    '''reformats the annotations into tracker-prediction format
    class_info is (class_ids, object_ids)
    returns as {object_id : box}'''
    # TODO: move reformatting into dataset
    assert len(class_info[0]) == len(boxes), f'class_info: {class_info} boxes: {boxes}'
    gt_boxes = {}
    for i in range(len(class_info[0])):
        if class_info[0][i] not in DESIRED_CLASSES:
            continue

        gt_boxes[class_info[1][i]] = boxes[i]

    return gt_boxes

def get_gt_masks(class_info : ((int,), (int,)), gt_mask) -> {int : {int : np.array}}:
    '''returns mask as class : object : map'''
    mask_maps = separate_segmentation_mask(gt_mask)

    assert len(class_info[1]) == len(mask_maps.keys()), f'keys are {class_info[1]} vs. {mask_maps.keys()}'

    gt_masks = {}

    for i in range(len(class_info[0])):
        curr_class = class_info[0][i]
        curr_obj = class_info[1][i]

        if curr_class not in gt_masks:
            gt_masks[curr_class] = {curr_obj : mask_maps[curr_obj]}
        else:
            gt_masks[curr_class][curr_obj] = mask_maps[curr_obj]

    return gt_masks

def get_gt_masks_as_pred(class_info : ((int,), (int,)), gt_mask : np.array) -> np.array:
    '''returns the given gt_mask as a 1 x 21 x W x H array'''
    x = np.zeros((1, 21, gt_mask.shape[0], gt_mask.shape[1]))

    for class_id, obj_id in zip(class_info[0], class_info[1]):
        x[0, class_id, :, :][gt_mask == obj_id] = 1

    return x

class Client:
    def _init_tracker(self):
        '''initializes mask or bb tracker'''
        if self.tracking:
            if self.run_type == 'BB':
                self.tracker = BoxTracker(self.logger)
            elif self.run_type == 'SM':
                # self.tracker = MaskTracker(self.logger)
                self.tracker = BoxMaskTracker(self.logger)
            else:
                raise NotImplementedError

    def _init_models(self, client_path = PARAMS['CLIENT_MODEL_PATH'], server_path = PARAMS['SERVER_MODEL_PATH']):
        '''initializes the model according to self.model_name'''
        self._refresh_iters = 1
        if self.is_gt():
            return

        self.logger.log_info('Creating models')

        if self.compressor == 'model':
            self.logger.log_debug(f'Setting up compression model {self.model_name}')
            student_model = get_student_model()
            if self.model_name == 'faster_rcnn':
                from split_models import model_2
                self.client_model = model_2.ClientModel(student_model)
                if self.server_connect:
                    self.server_model = nn.Identity()
                else:
                    self.server_model = model_2.ServerModel(student_model)

            elif self.model_name == 'deeplabv3':
                from split_models import model_3
                self.client_model = model_3.ClientModel(student_model)
                if self.server_connect:
                    self.server_model = nn.Identity()
                else:
                    self.server_model = model_3.ServerModel(student_model)

            else:
                raise NotImplementedError('No other model is implemented for splitting.')

            if client_path:
                self.client_model.load_state_dict(torch.load(client_path))

            if server_path and not self.server_connect:
                self.server_model.load_state_dict(torch.load(server_path))

            self.client_model.to(self.compressor_device)
            self.server_model.to(self.server_device)

        else: # classical compression
            if self.server_connect:
                self.logger.log_debug('Classical; connecting to server for detection.')
                pass

            else:
                self.logger.log_debug(
                    f'Classical compression; setting up model {self.model_name} for offline detection.')
                # offline; get models
                if self.model_name == 'faster_rcnn':
                    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
                    self.server_model = fasterrcnn_resnet50_fpn_v2(pretrained=True).eval()
                elif self.model_name == 'mask_rcnn':
                    from torchvision.models.detection import maskrcnn_resnet50_fpn
                    self.server_model = maskrcnn_resnet50_fpn(pretrained=True).eval()
                elif self.model_name == 'retinanet':
                    from torchvision.models.detection import retinanet_resnet50_fpn
                    self.server_model = retinanet_resnet50_fpn(pretrained=True).eval()
                elif self.model_name == 'deeplabv3':
                    from torchvision.models.segmentation import deeplabv3_resnet50
                    self.server_model = deeplabv3_resnet50(weights='DEFAULT').eval()
                else:
                    raise NotImplementedError

                self.server_model.to(self.server_device)

    def _init_multiprocessor(self):
        '''initializes multiprocessor objects if parallel_run
        calls init_mp in tracker'''
        if self.parallel_run:

            self.parallel_executor = ThreadPoolExecutor(max_workers=1)
            self.parallel_thread = None
            self.parallel_state = 0  # goes between ['rest', 'running', 'catchup']
            self.old_timestep = -1  # timestep tracker for other processes
            self.old_start_counter = -1  # counter to track the current video
            self.tracker.init_multiprocessing()
            self.old_class_map_detection = None

            if self.is_detection():
                # initialize shared detection objects needed for logging/eval
                self.parallel_thread = None
            elif self.is_segmentation():
                self.parallel_segmentor = None
            else:
                raise NotImplementedError('No other parallel runs.')

    def __init__(self, server_connect = PARAMS['USE_NETWORK'], run_type = PARAMS['RUN_TYPE'],
                 stats_log_dir = PARAMS['STATS_LOG_DIR'], dataset = PARAMS['DATASET'], tracking = PARAMS['TRACKING'],
                 task = PARAMS['TASK'], compressor = PARAMS['COMPRESSOR'],
                 refresh_type = PARAMS['BOX_REFRESH'], run_eval = PARAMS['EVAL'],
                 model_name = PARAMS['MODEL_NAME'], server_device = PARAMS['SERVER_DEVICE'],
                 compressor_device = PARAMS['COMPRESSOR_DEVICE'], socket_buffer_size = PARAMS['SOCK_BUFFER_SIZE'],
                 tracking_box_limit = PARAMS['BOX_LIMIT'], parallel_run = PARAMS['SMARTDET'],
                 detection_postprocessor : Callable = default_detection_postprocessor, log_stats = PARAMS['LOG_STATS']):

        self.socket, self.message, self.socket_buffer_size = None, None, socket_buffer_size
        self.logger, self.dataset = ConsoleLogger(), Dataset(dataset=dataset)
        self.stats_logger = DictionaryStatsLogger(logfile=f"{stats_log_dir}/client-{run_type}-{dataset}-{CURR_DATE}.log", log_stats=log_stats)
        self.server_connect, self.tracking, self.task, self.run_type = server_connect, tracking, task, run_type
        self.compressor, self.refresh_type = compressor, refresh_type
        self.run_eval, self.model_name, self.tracking_box_limit = run_eval, model_name, tracking_box_limit

        self.server_device, self.compressor_device = server_device, compressor_device
        self.parallel_run = parallel_run

        self.class_map_detection = {}

        self._init_tracker()
        self._init_models()
        self._init_multiprocessor()

        self.k = 1 # internal index counter
        self.start_counter = 0 # every time there is a new start to the vid, increase the counter
        self.detection_postprocess = detection_postprocessor

        if self.run_type == 'BB':
            self.get_gt = get_gt_detections
            self.get_gt_as_pred = get_gt_detections_as_pred
            self.get_pred = self.get_detections

        elif self.run_type == 'SM':
            self.get_gt = get_gt_masks
            self.get_gt_as_pred = get_gt_masks_as_pred
            self.get_pred = self.get_segmentation_mask

        else:
            raise NotImplementedError

    def is_detection(self):
        return self.task == 'det'

    def is_segmentation(self):
        return self.task == 'seg'

    def is_gt(self):
        return self.task == 'gt'

    def _connect(self, server_ip, server_port):
        assert self.server_connect
        self.logger.log_debug('Connecting from client')
        self.socket = socket(AF_INET, SOCK_STREAM)
        self.socket.connect((server_ip, server_port))
        self.socket.setsockopt(SOL_SOCKET, SO_SNDBUF, self.socket_buffer_size)
        self.logger.log_info('Successfully connected to socket')

    def _client_handshake(self):
        assert self.server_connect
        self.logger.log_debug('Sending handshake from client')
        self.socket.sendall(b'\00')
        ack = self.socket.recv(1)
        if(ack == b'\00'):
            self.logger.log_info('Successfully received server Handshake')
        else:
            self.logger.log_info('Message received not server ack')

    def start_client(self, server_ip, server_port):
        if self.server_connect:
            self._connect(server_ip, server_port)
            self._client_handshake()
            self.logger.log_info('Successfully started client')

        else:
            self.logger.log_info('Starting in offline mode.')

    def _send_encoded_data(self, data):
        '''if connected to the server, formats and sends the data
        else, simply store the data in self.message'''
        if self.server_connect:
            self.logger.log_debug('Sent encoded data to server.')
            data = pickle.dumps(data)
            length = pack('>Q', len(data))

            self.socket.sendall(length)
            self.socket.sendall(data)

            ack = self.socket.recv(1)
            self.logger.log_debug(f'Received server ack: {ack}.')

        else:
            self.message = data

    def _get_server_data(self):
        '''returns the server data in bytes'''
        assert self.server_connect
        self.logger.log_debug('Waiting for server data.')

        collected_message = False
        while not collected_message:
            bs = self.socket.recv(8)
            (length,) = unpack('>Q', bs)
            data = b''
            while len(data) < length:
                to_read = length - len(data)
                data += self.socket.recv(
                    4096 if to_read > 4096 else to_read)

            # send our 0 ack
            self.socket.sendall(b'\00')

            self.logger.log_debug('Received message.')

            return pickle.loads(data)

    def _get_model_data(self):
        '''uses the model (whether waiting for server or running offline) to get desired information'''

        assert self.is_detection() or self.is_segmentation()

        if self.server_connect: # connects to server and gets the data from there
            return self._get_server_data()

        else: # this part should be on the server.py for the online case
            self.logger.log_debug('Using offline model for detection.')

            client_data = self.message['data']
            if self.compressor == 'model':
                client_data_device = move_data_list_to_device(client_data, self.server_device)
                if self.model_name == 'faster_rcnn':
                    model_outputs = self.server_model(*client_data_device)[0]
                elif self.model_name == 'deeplabv3':
                    model_outputs = self.server_model(*client_data_device)
                else:
                    raise NotImplementedError('No specified detector model exists.')
            elif self.compressor == 'classical':
                decoded_frame = decode_frame(client_data)
                model_outputs = self.server_model([torch.from_numpy(decoded_frame).float().to(self.server_device)])[0] # first frame
            else:
                raise NotImplementedError('No other compression method exists.')

            self.logger.log_debug("Generated new BB with model (offline).")
            return move_data_dict_to_device(model_outputs, 'cpu')

    def _handle_server_data(self, server_data):
        '''Handles the server data, ie. formatting the model outputs into eval-friendly
        and tracker-friendly outputs'''
        if self.is_detection():
            # server_data is in the format of {'boxes' : [], 'labels' : [], 'scores' : []}
            detections, scores = map_coco_outputs(server_data)
            return detections
        elif self.is_segmentation():
            # server data is in the shape of 1 x 21 x W x H
            return server_data['out'].detach().numpy()
        else:
            raise NotImplementedError('No other server task other than detection.')

    def _compress_and_send_data(self, d):
        '''compresses information and sends it'''
        # offline case in model detection is handled in the individual helper functions
        self.logger.log_debug('Creating message for boxes.')
        # uses model for detection; requires some compression (make metrics)
        data, size_orig, class_info, gt, fname, _ = d

        if self.compressor == 'model':  # use a model (bottleneck) to compress it
            self.logger.log_debug('Performing compression using mdoel.')
            data_reshape = (data / 256).transpose((2, 0, 1))[np.newaxis, ...]

            # collect model runtime
            now = time.time()
            tensors_to_measure, other_info = self.client_model(
                torch.from_numpy(data_reshape).float().to(self.compressor_device))
            tensors_to_measure, other_info = move_data_list_to_device(tensors_to_measure, 'cpu'), \
                                             move_data_list_to_device(other_info, 'cpu')
            compression_time = time.time() - now

            size_compressed = get_tensor_size(tensors_to_measure)

            message = {'timestamp': time.time(), 'data': (*tensors_to_measure, *other_info)}

            self.stats_logger.push_log({'compressor': 'model'})

        elif self.compressor == 'classical':  # classical compression – compress data (frame) into jpg
            # TODO: make options – option to compress into jpg or compress video inside dataloader
            self.logger.log_debug('Performing classical compression on image.')

            # collect compression runtime
            before = time.time()
            encoded_frame = encode_frame(data)
            compression_time = time.time() - before
            size_compressed = sys.getsizeof(encoded_frame)

            message = {'timestamp': time.time(), 'data': encoded_frame}

            self.stats_logger.push_log({'compressor': 'classical'})

        else:
            raise NotImplementedError('No other compression method exists.')

        self.stats_logger.push_log({'encode_time': compression_time, 'message_size': size_compressed,
                                    'original_size': size_orig}, append=False)
        self.logger.log_info(f'Generated message with bytesize {size_compressed} and original {size_orig}')

        self._send_encoded_data(message)

    def _get_new_pred(self, d : (), before = None) -> {int : {int : (int,)}}:
        '''gets a new "accurate" bounding box/mask (from a separate detection pipeline)
        returns in the format of {object_id : (xyxy)}'''

        data, size_orig, class_info, gt, fname, _ = d
        gt_for_eval = self.get_gt_as_pred(class_info, gt)

        # if gt is used, function will simply return the ground truth;
        if self.is_gt():
            self.logger.log_debug('Using ground truth boxes.')
            # return ground truth boxes
            _, size_orig, class_info, gt, _, _ = d

            self.stats_logger.push_log({'gt' : True, 'original_size' : size_orig})
            return gt_for_eval

        # otherwise, use an actual detector
        self._compress_and_send_data(d)

        # response_time in the offline case will be the time it takes for the server model to run
        # in the online case it will be 2x latency + response_time
        now = time.time()
        server_data = self._get_model_data() # batch size 1
        if server_data is None:
            server_data = gt_for_eval

        self.stats_logger.push_log({'response_time': time.time() - now}, append=False)
        self.logger.log_info(f'Received information with response time {time.time() - now}')

        return self._handle_server_data(server_data)

    def check_detection_refresh(self, refresh_limit : int = PARAMS['REFRESH_ITERS']) -> bool:
        '''returns true if a refresh is needed, otherwise do nothing'''
        if self.refresh_type == 'fixed':
            if self._refresh_iters >= refresh_limit:
                self._refresh_iters = 1
                return True

            self._refresh_iters += 1
            return False

        else:
            raise NotImplementedError('Invalid Refresh Type')

    def _reinit_tracker(self, frame : np.ndarray, info : {int : (int,)}):
        '''creates a new tracker with the detections and frames
        detections is in the format of {object_id : [box]}'''
        if self.tracking:
            self.tracker.restart_tracker(frame, info)

    def get_tracker_pred_from_frame(self, frame) -> {int : (int,)}:
        '''updtes the tracker with the new frame and return the detections at that frame'''
        if self.tracking:
            now = time.time()
            info = self.tracker.process_frame(frame)
            self.stats_logger.push_log({'tracker': True, 'tracker_time': time.time() - now})
            return info

        return None

    def _reformat_detections_for_eval(self, gt_detections_with_classes, detections_with_classes):
        '''filters the detections by the ground truth detections and logs discrepancies'''
        # maps the ground truth labels with the predicted output (for evaluation purposes)
        class_map_detection, extra_detections = map_bbox_ids(detections_with_classes, gt_detections_with_classes)
        self.stats_logger.push_log({f'extra_class_{class_id}': extra_detection
                                    for class_id, extra_detection in extra_detections.items()},
                                   append=False)

        # remove the unnecessary boxes (don't bother tracking them)
        add_clause = lambda object_id: object_id in class_map_detection
        detections, _ = remove_classes_from_detections(detections_with_classes, add_clause=add_clause)

        self.logger.log_debug(f'Detections made: {len(detections)}, '
                              f'with mapping {class_map_detection}.')

        return detections, class_map_detection

    def get_detections(self, data, class_info, gt, d, now, start=False) -> ({int : [int]}, {int : int}):
        '''large function that gets the detections in the format of
        object_id : box, detection mapping to ground truth object IDs'''
        self.logger.log_debug('Re-generating bounding boxes.')

        gt_detections_with_classes = self.get_gt(class_info, gt)
        detections_with_classes = self._get_new_pred(d, now)

        if self.run_eval:
            self.logger.log_debug('Reformatting raw detections.')
            detections, class_map_detection = self._reformat_detections_for_eval(gt_detections_with_classes,
                                                                                 detections_with_classes)

        else:
            self.logger.log_debug('Reformatting and limiting raw detections (eval false).')
            return_clause = lambda _d: len(_d) > self.tracking_box_limit
            detections, class_map_detection = remove_classes_from_detections(detections_with_classes,
                                                                             return_clause=return_clause)

        return detections, class_map_detection

    def _reformat_masks_for_eval(self, gt_masks : {int : {int : np.array}}, masks : np.array) -> {int : np.array}:
        # TODO: use object_id instead of class_id
        '''filters the mask (1 x 21 x H x W) np array into a format of
        class_id : np.array mask'''
        self.logger.log_debug('Reformatting masks from straight array to dict')
        mask_classes = {}
        for class_id in gt_masks.keys():
            mask_classes[class_id] = masks[0, class_id, :, :]

        return mask_classes

    def get_segmentation_mask(self, data, class_info, gt, d, now, start=False) -> ({int: np.array}, {int : int}):
        '''gets the segmentation mask as a class_id : np array'''''
        self.logger.log_debug('Generating mask')

        gt_masks = self.get_gt(class_info, gt)
        masks = self._get_new_pred(d, now)

        # if self.run_eval:
        self.logger.log_debug('Reformatting raw masks.')
        reformatted_masks = self._reformat_masks_for_eval(gt_masks, masks)
        #
        # else:
        #     self.logger.log_debug('Limiting raw detections')

        class_map_detection = {}

        for k1, v1 in gt_masks.items():
            for k2 in v1.keys():
                class_map_detection[k2] = k1

        return reformatted_masks, class_map_detection

    def _execute_catchup(self, old_timestep : int, old_detections) -> (BoxTracker, {}):
        '''executes catchup on the tracker'''
        assert self.parallel_state == 2, 'States not synced; parallel state should be 2 (in catchup).'
        self.logger.log_debug('Executing catchup')
        if not self.tracking:
            self.logger.log_debug('No tracking; returning')
            return

        stats_logs = self.tracker.execute_catchup(old_timestep, old_detections)

        return stats_logs

    def handle_thread_result(self):
        '''information received from server; perform action with outdated information'''
        if self.is_detection(): # launch catchup algorithm
            old_detections, self.old_class_map_detection = self.parallel_thread.result()
            self.stats_logger.push_log({'num_detections': len(old_detections), 'tracker': False})
            self.logger.log_info('Parallel detections received; launching catchup')
            self.parallel_thread = self.parallel_executor.submit(self._execute_catchup, self.old_timestep,
                                                                 old_detections)
        elif self.is_segmentation():
            pass

    def handle_catchup_result(self):
        '''catchup completed; resync variables'''
        self.logger.log_info('Re-syncing self.tracker, handling catchup result.')
        self.class_map_detection = self.old_class_map_detection
        logs = self.parallel_thread.result()
        self.tracker.reset_mp()
        self.old_class_map_detection = None
        self.stats_logger.push_log(logs)
        self.stats_logger.push_log({'reset_tracker': True, 'reset_i': self.old_timestep, })

    def update_parallel_states(self, data):
        '''re-syncs variables and trackers if they are completed; performed in-thread'''
        if self.parallel_run:
            if not self.is_gt():
                if self.parallel_thread is not None: # if it's none, it's been completed
                    if self.parallel_thread.done():
                        if self.parallel_state == 1: # got bounding boxes for timestep i, time to execute catch-up
                            self.logger.log_info('Exiting state 1; going to state 2.')
                            if self.parallel_thread.exception():
                                err = self.parallel_thread.exception()
                                raise NotImplementedError(f'Detection thread errored for some reason: {str(err)}')

                            self.parallel_state = 2
                            self.handle_thread_result()

                        elif self.parallel_state == 2:
                            self.logger.log_debug('Catchup marked as completed; retrieving information.')
                            if self.parallel_thread.exception():
                                err = self.parallel_thread.exception()
                                raise NotImplementedError(f'Catchup thread errored for some reason: {str(err)}')
                            self.logger.log_debug('Successfully received catchup information.')
                            self.parallel_state = 0
                            self.parallel_thread = None

                    elif self.parallel_thread.running():
                        # do nothing if it is still running
                        pass
                    else:
                        raise NotImplementedError('No other status exists.')

                self.tracker.waiting_step(data)

        return None

    def eval_detections(self, gt_detections, pred_detections, class_map_detection):
        '''evaluates detections and pushes the logs'''
        self.logger.log_info('Evaluating detections.')
        pred_scores, missing_preds = eval_detections(gt_detections, pred_detections, class_map_detection)
        self.stats_logger.push_log({'missing_preds' : missing_preds, **pred_scores}, append=False)
        return

    def eval_segmentation(self, gt_mask, pred_mask, *kargs):
        # TODO: function
        self.logger.log_info('Evaluating segmentation.')
        pred_scores, missing_preds = eval_segmentation(gt_mask, pred_mask)
        self.stats_logger.push_log({'missing_preds' : missing_preds, **pred_scores}, append=False)

    def _reset_state_on_launch(self, data):
        self.parallel_state = 1
        self.old_timestep = self.k
        self.old_start_counter = self.start_counter
        self.stats_logger.push_log({'parallel_i': self.old_timestep})

    def launch_prediction(self, data, class_info, gt, d, now):
        '''creates a thread that gets the detections and submits it to threadpoolexecutor;
                also stores info of the detector'''
        if self.is_detection():
            self.logger.log_info('Launching parallel detection.')
        elif self.is_segmentation():
            self.logger.log_info('Launching parallel segmentation.')

        self.parallel_thread = self.parallel_executor.submit(self.get_pred, data, class_info, gt, d, now)
        self._reset_state_on_launch(data)

    def start_loop(self):
        try:
            end_of_previous_iteration = time.time()
            for i, d in enumerate(self.dataset.get_dataset()):
                # TODO: wrap into function for easier external testing
                self.logger.log_info(f'Starting iteration {i}')

                assert self.run_type in ('BB', 'SM'), 'No other type of run'

                data, size_orig, class_info, gt, fname, start = d
                self.stats_logger.push_log({'iter' : i, 'fname' : fname})

                # get the gt detections in {object : info} for eval
                gt_preds = self.get_gt(class_info, gt)
                if self.run_type == 'BB':
                    self.logger.log_debug(f'num gt_detections : {len(gt_preds)}')
                elif self.run_type == 'SM':
                    self.logger.log_debug(f'Got gt mask; type {type(gt_preds)}')

                # first check if the parallel detection process has completed
                self.update_parallel_states(data)

                if start: # start; no parallelization because tracker is not initialized
                    # TODO: handle starts with parallel_run better
                    self.k = 1
                    self.start_counter += 1
                    self.logger.log_info('Start of loop; initializing bounding box labels.')

                    pred, self.class_map_detection = self.get_pred(data, class_info, gt, d,
                                                                   end_of_previous_iteration,
                                                                   start=True)
                    self.stats_logger.push_log({'num_preds' : len(pred)})
                    self._reinit_tracker(data, pred)

                    self.stats_logger.push_log({'tracker': False})

                elif self.check_detection_refresh(): # get new bounding box, parallelize if applicable
                    # TODO: do nothing, delay detection refresh instead of AssertError
                    self.logger.log_info('Detection refresh criteria fulfilled – regenerating bbox.')
                    if self.parallel_run: # detections parallelized
                        assert self.parallel_state == 0, 'There is still a parallel process running at this time.'

                        # get detections for current timestamp
                        self.logger.log_debug('Using tracker for bounding boxes (executed detection separately).')
                        self.launch_prediction(data, class_info, gt, d, end_of_previous_iteration)
                        pred = self.get_tracker_pred_from_frame(data)

                    else: # detections in thread
                        pred, self.class_map_detection = self.get_pred(data, class_info, gt, d,
                                                                       end_of_previous_iteration,
                                                                       start=True)
                        self.stats_logger.push_log({'num_preds': len(pred)})
                        self._reinit_tracker(data, pred)

                        self.stats_logger.push_log({'tracker': False})

                else: # use tracker to get new bb
                    # detections here should be in the format of {object : bbox} as there is no
                    # class matching (inside class_map_detection)
                    self.logger.log_debug('Using tracker for bounding boxes.')
                    pred = self.get_tracker_pred_from_frame(data)

                if pred is None:
                    pred = gt_preds

                self.stats_logger.push_log({'iter_k': self.k})

                self.k += 1

                if self.run_eval:
                    if self.run_type == 'BB':
                        self.eval_detections(gt_preds, pred, self.class_map_detection)
                    elif self.run_type == 'SM':
                        self.eval_segmentation(gt_preds, pred)

                # push log
                self.stats_logger.push_log({}, append=True)

                # TODO: make postprocessing more flexible
                if self.run_type == 'BB':
                    self.detection_postprocess({'detections' : pred, 'gt_detections' : gt, 'frame' : data,
                                                'iteration' : self.k-1, 'mapping' : self.class_map_detection})
                end_of_previous_iteration = time.time()

        except Exception as ex:
            traceback.print_exc()
            self.logger.log_error(str(ex))

        finally:
            self.logger.log_info("Client loop ended.")
            self.close()

    def close(self):
        self.stats_logger.flush()

        if self.server_connect:
            self.socket.shutdown(SHUT_WR)
            self.socket.close()
            self.socket = None
        else:
            pass

        if self.parallel_run:
            self.parallel_executor.shutdown()

#main functionality for testing/debugging
if __name__ == '__main__':
    cp = Client()

    cp.start_client(PARAMS['HOST'], PARAMS['PORT'])
    cp.start_loop()