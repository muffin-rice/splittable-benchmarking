from socket import *
from struct import pack, unpack
import torch
from torch import nn

import pickle
import time
from params import PARAMS, CURR_DATE, DESIRED_CLASSES
from Logger import ConsoleLogger, DictionaryStatsLogger
from utils2 import *
from data import Dataset
from tracker import Tracker, BoxTracker, MaskTracker, BoxMaskTracker, segment_image_mrf
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

def create_input(data):
    return data

class Client:
    def _init_tracker(self):
        '''initializes mask or bb tracker'''
        if self.tracking:
            if self.run_type == 'BB':
                self.tracker = BoxTracker(self.logger)
            elif self.run_type == 'SM':
                # self.tracker = MaskTracker(self.logger)
                self.tracker = BoxMaskTracker(self.logger)
            elif self.run_type == 'BBSM':
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
                    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, \
                        FasterRCNN_ResNet50_FPN_V2_Weights
                    self.server_model = fasterrcnn_resnet50_fpn_v2(
                        weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT).eval()
                elif self.model_name == 'mask_rcnn':
                    from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
                    self.server_model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT).eval()
                elif self.model_name == 'retinanet':
                    from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
                    self.server_model = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT).eval()
                elif self.model_name == 'deeplabv3':
                    from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
                    self.server_model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT).eval()
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

    def __init__(self, server_connect = PARAMS['USE_NETWORK'], run_type = PARAMS['RUN_TYPE'],
                 stats_log_dir = PARAMS['STATS_LOG_DIR'], dataset = PARAMS['DATASET'], tracking = PARAMS['TRACKING'],
                 task = PARAMS['TASK'], compressor = PARAMS['COMPRESSOR'],
                 refresh_type = PARAMS['BOX_REFRESH'], run_eval = PARAMS['EVAL'],
                 model_name = PARAMS['MODEL_NAME'], server_device = PARAMS['SERVER_DEVICE'],
                 compressor_device = PARAMS['COMPRESSOR_DEVICE'], socket_buffer_read_size = PARAMS['SOCK_BUFFER_READ_SIZE'],
                 tracking_box_limit = PARAMS['BOX_LIMIT'], parallel_run = PARAMS['SMARTDET'],
                 detection_postprocessor : Callable = default_detection_postprocessor, log_stats = PARAMS['LOG_STATS']):

        self.socket, self.message, self.socket_buffer_read_size = None, None, socket_buffer_read_size
        self.logger, self.dataset = ConsoleLogger(), Dataset(dataset=dataset)
        self.stats_logger = DictionaryStatsLogger(logfile=f"{stats_log_dir}/client-{run_type}-{dataset}-{CURR_DATE}.log", log_stats=log_stats)
        self.server_connect, self.tracking, self.task, self.run_type = server_connect, tracking, task, run_type
        self.compressor, self.refresh_type = compressor, refresh_type
        self.run_eval, self.model_name, self.tracking_box_limit = run_eval, model_name, tracking_box_limit

        self.server_device, self.compressor_device = server_device, compressor_device
        self.parallel_run = parallel_run

        self.object_gt_mapping = {}

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

        elif self.run_type == 'BBSM':
            self.get_gt = get_gt_dets_from_mask
            self.get_gt_as_pred = get_gt_dets_from_mask_as_pred
            self.get_pred = self.get_detections

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
        self.socket.setsockopt(SOL_SOCKET, SO_SNDBUF, 10000)
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
                    self.socket_buffer_read_size if to_read > self.socket_buffer_read_size else to_read)

            # send our 0 ack
            self.socket.sendall(b'\00')

            self.logger.log_debug('Received message and sent ack.')

            return pickle.loads(data)

    def _get_model_data(self):
        '''uses the model (whether waiting for server or running offline) to get desired information'''

        assert not self.is_gt()

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
            return server_data['out'].detach().numpy()[0]
        else:
            raise NotImplementedError('No other server task other than detection.')

    def _compress_and_send_data(self, d):
        '''compresses information and sends it'''
        # offline case in model detection is handled in the individual helper functions
        self.logger.log_debug('Creating message for boxes.')
        # uses model for detection; requires some compression (make metrics)
        data, size_orig, class_info, gt, fname, _ = d

        if self.compressor == 'model':  # use a model (bottleneck) to compress it
            self.logger.log_debug('Performing compression using model.')
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
        gt_as_pred = self.get_gt_as_pred(class_info, gt)

        # if gt is used, function will simply return the ground truth;
        if self.is_gt():
            self.logger.log_debug('Using ground truth boxes.')
            # return ground truth boxes
            _, size_orig, class_info, gt, _, _ = d

            self.stats_logger.push_log({'gt' : True, 'original_size' : size_orig})
            return gt_as_pred

        # otherwise, use an actual detector
        self._compress_and_send_data(d)

        # response_time in the offline case will be the time it takes for the server model to run
        # in the online case it will be 2x latency + response_time
        now = time.time()
        server_data = self._get_model_data() # batch size 1
        if server_data is None:
            server_data = gt_as_pred

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
        '''filters the detections by the ground truth detections and logs discrepancies
        returns detections, object mapping (model_id : gt_id)'''
        # maps the ground truth labels with the predicted output (for evaluation purposes)
        object_id_mapping, extra_detections = map_bbox_ids(detections_with_classes, gt_detections_with_classes)
        self.stats_logger.push_log({f'extra_class_{class_id}': extra_detection
                                    for class_id, extra_detection in extra_detections.items()},
                                   append=False)

        # remove the unnecessary boxes (don't bother tracking them)
        add_clause = lambda object_id: object_id in object_id_mapping
        return_clause = lambda _d: len(_d) > self.tracking_box_limit
        detections, _ = remove_classes_from_pred(detections_with_classes, add_clause=add_clause,
                                                 return_clause=return_clause)

        self.logger.log_debug(f'Detections made: {len(detections)}, with mapping {object_id_mapping}.')

        return detections, object_id_mapping

    def get_detections(self, data, class_info, gt, d, now, start=False) -> ({int : [int]}, {int : int}):
        '''large function that gets the detections in the format of
        object_id : box, detection mapping to ground truth object IDs'''
        self.logger.log_debug('Re-generating bounding boxes.')

        gt_detections_with_classes = self.get_gt(class_info, gt)
        detections_with_classes = self._get_new_pred(d, now)

        if self.run_eval:
            self.logger.log_debug('Reformatting raw detections.')
            detections, object_id_mapping = self._reformat_detections_for_eval(gt_detections_with_classes,
                                                                                 detections_with_classes)

        else:
            self.logger.log_debug('Reformatting and limiting raw detections (eval false).')
            return_clause = lambda _d: len(_d) > self.tracking_box_limit
            detections, object_id_mapping = remove_classes_from_pred(detections_with_classes,
                                                                     return_clause=return_clause)

        return detections, object_id_mapping

    def _reformat_masks_for_eval(self, gt_masks_with_classes : {int : {int : np.array}},
                                 pred_masks_with_classes : np.array) -> ({int : np.array}, {int : int}):
        '''filters the masks by the ground truth detections and logs discrepancies
                returns detections, object mapping (model_id : gt_id)'''
        # make the 21 x H x W mask into a dictionary and also binarized
        binarized_pred_masks = binarize_mask(pred_masks_with_classes)
        pred_masks_with_classes = {i : binarized_pred_masks[i] for i in range(binarized_pred_masks.shape[0])}
        # maps the ground truth labels with the predicted output (for evaluation purposes)
        object_id_mapping, extra_masks = map_mask_ids(pred_masks_with_classes, gt_masks_with_classes)
        self.stats_logger.push_log({f'extra_class_{class_id}': extra_detection
                                    for class_id, extra_detection in extra_masks.items()},
                                   append=False)

        # remove the unnecessary boxes (don't bother tracking them)
        add_clause = lambda object_id: object_id in object_id_mapping
        reformatted_masks, _ = remove_classes_from_pred(pred_masks_with_classes, add_clause=add_clause)

        self.logger.log_debug(f'Detections made: {len(reformatted_masks)}, with mapping {object_id_mapping}.')

        return reformatted_masks, object_id_mapping

    def get_segmentation_mask(self, data, class_info, gt, d, now, start=False) -> ({int: np.array}, {int : int}):
        '''gets the segmentation mask as a class_id : np array'''''
        self.logger.log_debug('Generating mask')

        gt_masks = self.get_gt(class_info, gt)
        masks = self._get_new_pred(d, now)

        # if self.run_eval:
        self.logger.log_debug('Reformatting raw masks.')
        reformatted_masks, object_id_mapping = self._reformat_masks_for_eval(gt_masks, masks)
        #
        # else:
        #     self.logger.log_debug('Limiting raw detections')

        return reformatted_masks, object_id_mapping

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
        if not self.is_gt(): # launch catchup algorithm
            old_pred, self.old_object_gt_mapping = self.parallel_thread.result()
            self.stats_logger.push_log({'num_detections': len(old_pred), 'tracker': False})
            self.logger.log_info('Parallel predictions received; launching catchup')
            self.parallel_thread = self.parallel_executor.submit(self._execute_catchup, self.old_timestep,
                                                                 old_pred)


    def handle_catchup_result(self):
        '''catchup completed; resync variables'''
        self.logger.log_info('Re-syncing self.tracker, handling catchup result.')
        self.object_gt_mapping = self.old_object_gt_mapping
        logs = self.parallel_thread.result()
        self.tracker.reset_mp()
        self.old_object_gt_mapping = None
        self.stats_logger.push_log(logs)
        self.stats_logger.push_log({'reset_tracker': True, 'reset_i': self.old_timestep, })

    def update_parallel_states(self, data):
        '''re-syncs variables and trackers if they are completed; performed in-thread'''
        if self.parallel_run:
            if not self.is_gt():
                if self.parallel_thread is not None: # if it's none, it's been completed
                    self.tracker.waiting_step(data)

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
                                self.logger.log_error('Catchup failed for some reason; quitting')
                                traceback.print_exc()
                                err = self.parallel_thread.exception()
                                raise err
                            self.logger.log_debug('Successfully received catchup information.')
                            self.handle_catchup_result()
                            self.parallel_state = 0
                            self.parallel_thread = None

                            return

                    elif self.parallel_thread.running():
                        # do nothing if it is still running
                        pass
                    else:
                        raise NotImplementedError('No other status exists.')

        return None

    def close_mp(self):
        self.logger.log_info('Closing the other threads')
        if self.parallel_run:
            if self.parallel_state != 0: # idle
                assert self.parallel_thread is not None
                self.parallel_thread.cancel()

            self.tracker.close_mp()
            self.parallel_state = 0
            self.parallel_thread = None

    def eval_detections(self, gt_detections, pred_detections, object_id_mapping):
        '''evaluates detections and pushes the logs'''
        # TODO: Combine eval functions
        self.logger.log_info('Evaluating detections.')
        pred_scores, missing_preds = eval_detections(gt_detections, pred_detections, object_id_mapping)
        self.stats_logger.push_log({'missing_preds' : missing_preds, **pred_scores}, append=False)
        return

    def eval_segmentation(self, gt_mask, pred_mask, object_id_mapping):
        self.logger.log_info('Evaluating segmentation.')
        pred_scores, missing_preds = eval_segmentation(gt_mask, pred_mask, object_id_mapping)
        self.stats_logger.push_log({'missing_preds' : missing_preds, **pred_scores}, append=False)

    def _reset_state_on_launch(self, data):
        '''state update for post-function of launching parallel detection'''
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

    def start_loop(self, start_i = 0):
        try:
            end_of_previous_iteration = time.time()
            for i, d in enumerate(self.dataset.get_dataset()):
                start_time_of_iteration = time.time()
                i = i + start_i
                # TODO: wrap into function for easier external testing
                self.logger.log_info(f'Starting iteration {i}')

                assert self.run_type in ('BB', 'SM', 'BBSM'), 'No other type of run'

                data, size_orig, class_info, gt, fname, start = d
                self.stats_logger.push_log({'iter' : i, 'fname' : fname})

                if start:
                    self.close_mp() # should be done at "end" but do it here

                # get the gt detections in {object : info} for eval
                gt_preds = self.get_gt(class_info, gt) # class : {object : info}
                gt_as_pred = self.get_gt_as_pred(class_info, gt) # {object : info}
                if self.run_type == 'BB':
                    self.logger.log_debug(f'num gt_detections : {len(gt_as_pred)}')
                elif self.run_type == 'SM' or self.run_type == 'BBSM':
                    self.logger.log_debug(f'Got gt mask; type {type(gt_preds)} with len {len(gt_preds)}')

                # first check if the parallel detection process has completed
                self.update_parallel_states(data)

                if start: # start; no parallelization because tracker is not initialized
                    self.k = 1
                    self.start_counter += 1
                    self.logger.log_info(f'Start of loop ({fname}); initializing bounding box labels.')

                    pred, self.object_gt_mapping = self.get_pred(data, class_info, gt, d,
                                                                 end_of_previous_iteration,
                                                                 start=True)
                    self.stats_logger.push_log({'num_preds' : len(pred)})
                    self._reinit_tracker(data, pred)

                    self.stats_logger.push_log({'tracker': False, 'start' : True})

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
                        pred, self.object_gt_mapping = self.get_pred(data, class_info, gt, d,
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
                    pred = gt_as_pred

                if len(pred) == 0:
                    self.logger.log_info('There are no predictions for this frame.')

                self.stats_logger.push_log({'iter_k': self.k})

                self.k += 1

                time_to_eval_start = time.time()
                if self.run_eval:
                    if self.run_type == 'BB':
                        self.eval_detections(gt_as_pred, pred, self.object_gt_mapping)
                    elif self.run_type == 'SM':
                        self.eval_segmentation(gt_as_pred, pred, self.object_gt_mapping)
                    elif self.run_type == 'BBSM': # pred is still in "bb" mode
                        self.logger.log_debug('Eval detections and pred')
                        self.eval_detections(gt_as_pred, pred, self.object_gt_mapping)
                        gt_masks_as_pred = get_gt_masks_as_pred(class_info, gt)
                        # map boxes to integers
                        pred_masks = self.tracker.get_masks()
                        self.eval_segmentation(gt_masks_as_pred, pred_masks, self.object_gt_mapping)

                now = time.time()
                self.stats_logger.push_log({'iteration_time' : now - start_time_of_iteration,
                                            'time_to_eval' : now - time_to_eval_start})
                # push log
                self.stats_logger.push_log({}, append=True)

                # TODO: make postprocessing more flexible
                if self.run_type == 'BB':
                    self.detection_postprocess({'detections' : pred, 'gt_detections' : gt, 'frame' : data,
                                                'iteration' : self.k-1, 'mapping' : self.object_gt_mapping})
                end_of_previous_iteration = time.time()

        except Exception as ex:
            traceback.print_exc()
            self.logger.log_error(str(ex))
            self.close()

        finally:
            # check if any processes still remain in the parallel
            self.close_mp()
            self.logger.log_info("Client loop ended.")

    def start_loop2(self, file_batch_size = PARAMS['FILE_BATCH_SIZE']):
        if not PARAMS['FILE_TRANSFER']:
            self.start_loop()
            return

        # transfer files back and forth
        with open(PARAMS['DATA_PICKLE_FILES'], 'rb') as f:
            all_server_fnames = pickle.load(f)

        # start_i = 0
        data_dir = PARAMS['DATA_DIR']

        for fname_batch in all_server_fnames:
            self._send_encoded_data(fname_batch)
            self.logger.log_info(f'Sent first batch with {len(fname_batch)} files')
            byte_files = self._get_server_data()
            self.logger.log_info('Received batch of files; processing')

            assert len(fname_batch) == len(byte_files)

            for fname, byte_file in zip(fname_batch, byte_files):
                # os.makedirs(f'{data_dir}/{fname}', exist_ok=True)
                with open(f'{data_dir}/{fname}', 'wb') as f:
                    f.write(byte_file)

            self.logger.log_debug('Files received and written; starting loop')
            self.start_loop(start_i = 0)
            # batch fname size fixed at 5
            # start_i += file_batch_size

            # remove the files
            self.logger.log_debug('Removing files')
            for fname in fname_batch:
                os.remove(f'{data_dir}/{fname}')


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
    cp.start_loop2()
    cp.close()