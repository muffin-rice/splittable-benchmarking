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
from split_models import model_2
from tracker import Tracker
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from torchdistill.common import yaml_util

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
    assert len(class_info[0]) == len(boxes), f'class_info: {class_info} boxes: {boxes}'
    gt_boxes = {}
    for i in range(len(class_info[0])):
        if class_info[0][i] not in DESIRED_CLASSES:
            continue

        gt_boxes[class_info[1][i]] = boxes[i]

    return gt_boxes

class Client:
    def _init_tracker(self):
        if self.tracking:
            self.tracker = Tracker()

    def _init_detector(self):
        self._refresh_iters = 1
        if not self.detection:
            return

        if self.detection_compression == 'model':
            # use student model
            self.logger.log_debug(f'Setting up compression model {self.detector_model}.')

            if self.detector_model == 'faster_rcnn':
                student_model = get_student_model(PARAMS['FASTER_RCNN_YAML'])
                self.client_model = model_2.ClientModel(student_model)
                if self.server_connect:
                    self.server_model = nn.Identity()
                else:
                    self.server_model = model_2.ServerModel(student_model)

                self.client_model.to(self.compressor_device)
                self.server_model.to(self.detector_device)
            else:
                raise NotImplementedError

        else: # classical compression
            if self.server_connect:
                self.logger.log_debug('Classical; connecting to server for detection.')
                pass
            else:
                self.logger.log_debug(f'Classical compression; setting up model {self.detector_model} for offline detection.')
                # offline; get models
                if self.detector_model == 'faster_rcnn':
                    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
                    self.server_model = fasterrcnn_resnet50_fpn_v2(pretrained=True).eval()
                elif self.detector_model == 'mask_rcnn':
                    from torchvision.models.detection import maskrcnn_resnet50_fpn
                    self.server_model = maskrcnn_resnet50_fpn(pretrained=True).eval()
                elif self.detector_model == 'retinanet':
                    from torchvision.models.detection import retinanet_resnet50_fpn
                    self.server_model = retinanet_resnet50_fpn(pretrained=True).eval()
                else:
                    raise NotImplementedError

                self.server_model.to(self.detector_device)

    def _init_multiprocessor(self):
        if self.parallel_run:
            assert self.detection, 'Cannot parallelize detection when detection is false.'
            self.parallel_executor = ThreadPoolExecutor(max_workers=1)
            # initialize shared detection objects needed for logging/eval
            self.parallel_detector = None
            # self.server_model.to(self.detector_device)
            self.parallel_state = 0 # goes between ['rest', 'running', 'catchup']
            self.old_timestep = -1 # timestep tracker for other processes
            self.old_start_counter = -1 # counter to track the current video
            self.old_frame = None # frame from original
            self.old_class_map_detection = None
            self.catchup_frames = [] # list of images for catch-up to use

    def __init__(self, server_connect = PARAMS['USE_NETWORK'], run_type = PARAMS['RUN_TYPE'],
                 stats_log_dir = PARAMS['STATS_LOG_DIR'], dataset = PARAMS['DATASET'], tracking = PARAMS['TRACKING'],
                 detection = PARAMS['DETECTION'], detection_compression = PARAMS['DET_COMPRESSOR'],
                 refresh_type = PARAMS['BOX_REFRESH'], run_eval = PARAMS['EVAL'],
                 detector_model = PARAMS['DETECTOR_MODEL'], detector_device = PARAMS['DETECTION_DEVICE'],
                 compressor_device = PARAMS['COMPRESSOR_DEVICE'], socket_buffer_size = PARAMS['SOCK_BUFFER_SIZE'],
                 tracking_box_limit = PARAMS['BOX_LIMIT'], parallel_run = PARAMS['DET_PARALLEL'],
                 detection_postprocessor : Callable = default_detection_postprocessor, log_stats = PARAMS['LOG_STATS']):

        self.socket, self.message, self.socket_buffer_size = None, None, socket_buffer_size
        self.logger, self.dataset, self.stats_logger = ConsoleLogger(), Dataset(dataset=dataset), \
                                                       DictionaryStatsLogger(logfile=f"{stats_log_dir}/client-{dataset}-{CURR_DATE}.log", log_stats=log_stats)
        self.server_connect, self.tracking, self.detection, self.run_type = server_connect, tracking, detection, run_type
        self.detection_compression, self.refresh_type = detection_compression, refresh_type
        self.run_eval, self.detector_model, self.tracking_box_limit = run_eval, detector_model, tracking_box_limit

        self.detector_device, self.compressor_device = detector_device, compressor_device
        self.parallel_run = parallel_run

        self.class_map_detection = {}

        self._init_tracker()
        self._init_detector()
        self._init_multiprocessor()

        self.k = 1 # internal index counter
        self.start_counter = 0 # every time there is a new start to the vid, increase the counter
        self.detection_postprocess = detection_postprocessor

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

    def start(self, server_ip, server_port):
        if self.server_connect:
            self._connect(server_ip, server_port)
            self._client_handshake()
            self.logger.log_info('Successfully started client')

        else:
            self.logger.log_info('Starting in offline mode.')

    def _send_encoder_data(self, data):
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
        '''returns the server data in any format'''
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

    def _get_model_bb(self):
        '''uses the model to get the bounding box
        will never be called if DETECTOR is False'''

        assert self.detection

        if self.server_connect: # connects to server and gets the data from there
            return self._get_server_data()

        else: # this part should be on the server.py for the online case
            self.logger.log_debug('Using offline model for detection.')

            client_data = self.message['data']
            if self.detection_compression == 'model':
                client_data_device = move_data_list_to_device(client_data, self.detector_device)
                if self.detector_model == 'faster_rcnn':
                    model_outputs = self.server_model(*client_data_device)[0]
                else:
                    raise NotImplementedError('No specified detector model exists.')
            elif self.detection_compression == 'classical':
                decoded_frame = decode_frame(client_data)
                model_outputs = self.server_model([torch.from_numpy(decoded_frame).float().to(self.detector_device)])[0] # first frame
            else:
                raise NotImplementedError('No other compression method exists.')

            self.logger.log_debug("Generated new BB with model (offline).")
            return move_data_dict_to_device(model_outputs, 'cpu')

    def _handle_server_data(self, server_data, server_model = PARAMS['DETECTOR_MODEL']):
        '''Handles the server data, ie. formatting the model outputs into eval-friendly
        and tracker-friendly outputs'''
        if self.detection:
            # server_data is in the format of {'boxes' : [], 'labels' : [], 'scores' : []}
            detections, scores = map_coco_outputs(server_data)
            return detections
        else:
            raise NotImplementedError('No other server task other than detection.')

    def _get_new_bounding_box(self, d : (), before = None) -> {int : {int : (int,)}}:
        '''gets a new "accurate" bounding box (from a separate detection pipeline)
        returns in the format of {object_id : (xyxy)}'''

        data, size_orig, class_info, gt, fname, _ = d
        gt_detections = get_gt_detections(class_info, gt)

        # if gt is used, get bounding box will simply return the ground truth
        if not self.detection:
            self.logger.log_debug('Using ground truth boxes.')
            # return ground truth boxes
            _, size_orig, class_info, gt, _, _ = d

            self.stats_logger.push_log({'gt' : True, 'original_size' : size_orig})
            return gt_detections

        # otherwise, use an actual detector
        # offline case in model detection is handled in the individual helper functions
        self.logger.log_debug('Creating message for boxes.')
        # uses model for detection; requires some compression (make metrics)
        if self.detection_compression == 'model':  # use a model (bottleneck) to compress it
            self.logger.log_debug('Performing compression using mdoel.')
            data, size_orig, class_info, gt, fname, _ = d
            data_reshape = (data/256).transpose((2,0,1))[np.newaxis, ...]

            # collect model runtime
            now = time.time()
            tensors_to_measure, other_info = self.client_model(torch.from_numpy(data_reshape).float().to(self.compressor_device))
            tensors_to_measure, other_info = move_data_list_to_device(tensors_to_measure, 'cpu'), move_data_list_to_device(other_info, 'cpu')
            compression_time = time.time() - now

            size_compressed = get_tensor_size(tensors_to_measure)

            message = {'timestamp': time.time(), 'data': (*tensors_to_measure, *other_info)}

            self.stats_logger.push_log({'compressor' : 'model'})

        elif self.detection_compression == 'classical':  # classical compression – compress data (frame) into jpg
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

        self._send_encoder_data(message)

        # response_time in the offline case will be the time it takes for the server model to run
        # in the online case it will be 2x latency + response_time
        now = time.time()
        server_data = self._get_model_bb() # batch size 1
        if server_data is None:
            server_data = gt_detections

        self.stats_logger.push_log({'response_time': time.time() - now}, append=False)
        self.logger.log_info(f'Received detection with response time {time.time() - now}')

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

    def _reinit_tracker(self, frame : np.ndarray, detections : {int : (int,)}):
        '''creates a new tracker with the detections and frames
        detections is in the format of {object_id : [box]}'''
        if self.tracking:
            self.tracker.handle_new_detection(frame, detections)

    def get_tracker_bounding_box(self, frame) -> {int : (int,)}:
        '''updtes the tracker with the new frame and return the detections at that frame'''
        if self.tracking:
            now = time.time()
            detections = self.tracker.update(frame)
            self.stats_logger.push_log({'tracker': True, 'tracker_time': time.time() - now})
            return detections

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
        '''large function that gets the ground truth detections in correct format'''
        self.logger.log_debug('Re-generating bounding boxes.')

        gt_detections_with_classes = get_gt_detections(class_info, gt)
        detections_with_classes = self._get_new_bounding_box(d, now)

        if self.run_eval:
            self.logger.log_debug('Reformatting raw detections.')
            detections, class_map_detection = self._reformat_detections_for_eval(gt_detections_with_classes,
                                                                                 detections_with_classes)

        else:
            self.logger.log_debug('Reformating and limiting raw detections (eval false).')
            return_clause = lambda _d: len(_d) > self.tracking_box_limit
            detections, class_map_detection = remove_classes_from_detections(detections_with_classes,
                                                                             return_clause=return_clause)

        return detections, class_map_detection

    def _execute_catchup(self, old_timestep : int, old_detections, starting_frame) -> (Tracker, {}):
        '''iterates until self.catchup_list is empty, returns new tracker with updated start and logs'''
        assert self.parallel_state == 2, 'States not synced; parallel state should be 2 (in catchup).'
        if not self.tracking:
            self.logger.log_debug('No tracking; returning')
            return

        stats_logs = {}

        starting_length = len(self.catchup_frames)
        self.logger.log_debug(f'Starting from {old_timestep}, processing {starting_length}')
        num_processed = 0
        # from _init_trackers()
        catchup_tracker = Tracker()
        catchup_tracker.handle_new_detection(starting_frame, old_detections)
        curr_detections = old_detections # do nothing with old detections
        starting_time = time.time()
        while num_processed < len(self.catchup_frames):
            if num_processed > 10:
                raise AssertionError('Catchup taking too many iterations')

            curr_detections = catchup_tracker.update(self.catchup_frames[num_processed])
            self.logger.log_debug(f'Processed frame {num_processed}')
            num_processed += 1

        stats_logs['process_time'] = time.time() - starting_time
        stats_logs['added_frames'] = len(self.catchup_frames) - starting_length

        return catchup_tracker, stats_logs

    def update_parallel_states(self, data):
        '''re-syncs variables and trackers if they are completed; this is in-thread'''
        if self.parallel_run:
            if self.parallel_detector is not None: # if it's none, it's been completed
                if self.parallel_detector.done():
                    if self.parallel_state == 1: # got bounding boxes for timestep i, time to execute catch-up
                        self.logger.log_info('Exiting state 1; going to state 2.')
                        if self.parallel_detector.exception():
                            err = self.parallel_detector.exception()
                            raise NotImplementedError(f'Detection thread errored for some reason: {str(err)}')

                        self.parallel_state = 2
                        old_detections, self.old_class_map_detection = self.parallel_detector.result()
                        self.stats_logger.push_log({'num_detections': len(old_detections), 'tracker': False})
                        self.logger.log_info('Parallel detections received; launching catchup')
                        self.parallel_detector = self.parallel_executor.submit(self._execute_catchup, self.old_timestep,
                                                                               old_detections, self.old_frame)

                    elif self.parallel_state == 2:
                        self.logger.log_debug('Catchup marked as completed; retrieving information.')
                        if self.parallel_detector.exception():
                            err = self.parallel_detector.exception()
                            raise NotImplementedError(f'Catchup thread errored for some reason: {str(err)}')
                        self.logger.log_debug('Successfully received catchup information.')
                        self.parallel_state = 0
                        if self.tracking:
                            self.logger.log_info('Re-setting self.tracker.')
                            self.class_map_detection = self.old_class_map_detection
                            new_tracker, logs = self.parallel_detector.result()
                            self.tracker = new_tracker
                            self.old_class_map_detection = None
                            self.stats_logger.push_log(logs)
                            self.stats_logger.push_log({'reset_tracker' : True, 'reset_i' : self.old_timestep,})

                        self.parallel_detector = None

                elif self.parallel_detector.running():
                    # do nothing if it is still running
                    pass
                else:
                    raise NotImplementedError('No other status exists.')

            self.catchup_frames.append(data) # append data after retrieving new information

        return None

    def eval_detections(self, gt_detections, pred_detections, class_map_detection):
        '''evaluates detections and pushes the logs'''
        self.logger.log_info('Evaluating detections.')
        pred_scores, missing_preds = eval_detections(gt_detections, pred_detections, class_map_detection)
        self.stats_logger.push_log({'missing_preds' : missing_preds, **pred_scores}, append=False)
        return

    def launch_detection(self, data, class_info, gt, d, now):
        '''creates a thread that gets the detections and submits it to threadpoolexecutor;
        also stores info of the detector'''
        self.logger.log_info('Launching parallel detection.')
        self.parallel_detector = self.parallel_executor.submit(self.get_detections, data, class_info, gt, d, now)
        self.parallel_state = 1
        self.old_frame = data
        self.old_timestep = self.k
        self.old_start_counter = self.start_counter
        self.catchup_frames = []
        self.stats_logger.push_log({'parallel_i': self.old_timestep})

    def start_loop(self):
        try:
            end_of_previous_iteration = time.time()
            for i, d in enumerate(self.dataset.get_dataset()):
                # TODO: wrap into function for easier external testing
                self.logger.log_info(f'Starting iteration {i}')

                if self.run_type == 'BB':

                    data, size_orig, class_info, gt, fname, start = d
                    self.stats_logger.push_log({'iter' : i, 'fname' : fname})

                    # get the gt detections in {object : bbox} for eval
                    gt_detections = get_gt_detections_as_pred(class_info, gt)
                    self.logger.log_debug(f'num gt_detections : {len(gt_detections)}')

                    # first check if the parallel detection process has completed
                    self.update_parallel_states(data)

                    if start: # start; no parallelization because tracker is not initialized
                        # TODO: handle starts with parallel_run better
                        self.k = 1
                        self.start_counter += 1
                        self.logger.log_info('Start of loop; initializing bounding box labels.')
                        detections, self.class_map_detection = self.get_detections(data, class_info, gt, d,
                                                                                   end_of_previous_iteration, start=True)
                        # log number of detections (useful for eval tracker time)
                        self.stats_logger.push_log({'num_detections': len(detections), 'tracker' : False})
                        self._reinit_tracker(data, detections)

                    elif self.check_detection_refresh(): # get new bounding box, parallelize if applicable
                        # TODO: do nothing, delay detection refresh instead of AssertError
                        self.logger.log_info('Detection refresh criteria fulfilled – regenerating bbox.')
                        if self.parallel_run: # detections parallelized
                            assert self.parallel_state == 0, 'There is still a parallel process running at this time.'
                            self.launch_detection(data, class_info, gt, d, end_of_previous_iteration)

                            # get detections for current timestamp
                            self.logger.log_debug('Using tracker for bounding boxes (executed detection separately).')
                            detections = self.get_tracker_bounding_box(data)
                            self.stats_logger.push_log({'num_detections': len(detections), 'tracker': True})
                            if detections is None:
                                detections = gt_detections

                        else: # detections in thread
                            detections, self.class_map_detection = self.get_detections(data, class_info, gt, d,
                                                                                       end_of_previous_iteration)
                            self._reinit_tracker(data, detections)

                    else: # use tracker to get new bb
                        # detections here should be in the format of {object : bbox} as there is no
                        # class matching (inside class_map_detection)
                        self.logger.log_debug('Using tracker for bounding boxes.')
                        detections = self.get_tracker_bounding_box(data)
                        if detections is None:
                            detections = gt_detections

                    self.stats_logger.push_log({'iter_k': self.k})

                    self.k += 1

                    if self.run_eval:
                        self.eval_detections(gt_detections, detections, self.class_map_detection)

                else:
                    raise NotImplementedError('No other type of run besides bounding box')

                # push log
                self.stats_logger.push_log({}, append=True)

                # TODO: make postprocessing more flexible
                self.detection_postprocess({'detections' : detections, 'gt_detections' : gt_detections, 'frame' : data,
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

def get_student_model(yaml_file = PARAMS['FASTER_RCNN_YAML']):
    if yaml_file is None:
        return None

    config = yaml_util.load_yaml_file(os.path.expanduser(yaml_file))
    models_config = config['models']
    student_model_config = models_config['student_model'] if 'student_model' in models_config else models_config[
        'model']
    student_model = load_model(student_model_config, PARAMS['DETECTION_DEVICE']).eval()

    return student_model

#main functionality for testing/debugging
if __name__ == '__main__':
    cp = Client()

    cp.start(PARAMS['HOST'], PARAMS['PORT'])
    cp.start_loop()