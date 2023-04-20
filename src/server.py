from multiprocessing import connection
import pickle
import time
import sys
from socket import *
from struct import pack, unpack

import torch

from params import PARAMS, CURR_DATE
from Logger import ConsoleLogger, DictionaryStatsLogger
from utils2 import *
import traceback

class Server:
    '''Class for server operations. No functionality for offline evaluation (server does not do any eval).'''

    def _init_model(self, server_path = PARAMS['SERVER_MODEL_PATH']):
        if self.compressor == 'model':
            self.logger.log_info(f'Setting up model {self.model_name}')
            student_model = get_student_model()

            if self.model_name == 'faster_rcnn':
                from split_models.model_2 import ServerModel
                self.server_model = ServerModel(student_model)
            elif self.model_name == 'deeplabv3':
                from split_models.model_3 import ServerModel
                self.server_model = ServerModel(student_model)
            else:
                raise NotImplementedError('No other models have been implemented yet.')

            if server_path:
                self.server_model.load_state_dict(torch.load(server_path))

        elif self.compressor == 'classical':
            if self.model_name == 'faster_rcnn':
                from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
                self.server_model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT).eval()
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

        pass

    def __init__(self, server_connect = PARAMS['USE_NETWORK'], compressor = PARAMS['COMPRESSOR'],
                 model_name = PARAMS['MODEL_NAME'], server_device = PARAMS['SERVER_DEVICE'],
                 socket_buffer_read_size = PARAMS['SOCK_BUFFER_READ_SIZE'], log_stats = PARAMS['LOG_STATS'],
                 task = PARAMS['TASK'], stats_log_dir = PARAMS['STATS_LOG_DIR'], run_type = PARAMS['RUN_TYPE'],
                 dataset = PARAMS['DATASET']):
        self.socket, self.connection, self.server_connect, self.socket_buffer_read_size = None, None, server_connect, socket_buffer_read_size
        self.data, self.logger = None, ConsoleLogger()
        dict_logfile = f"{stats_log_dir}/server-{run_type}-{dataset}-{CURR_DATE}.log"
        self.stats_logger = DictionaryStatsLogger(logfile=dict_logfile, log_stats=log_stats)
        self.logger.log_info(f'Writing dictionary to {dict_logfile}')
        self.task = task
        self.compressor, self.model_name, self.server_device = compressor, model_name, server_device
        self._init_model()

    def is_detection(self):
        return self.task == 'det'

    def is_segmentation(self):
        return self.task == 'seg'

    def listen(self, server_ip, server_port):
        self.socket = socket(AF_INET, SOCK_STREAM)
        self.socket.setsockopt(SOL_SOCKET, SO_SNDBUF, 10000)
        self.logger.log_info(f"Binding to {server_ip}:{server_port}")
        self.socket.bind((server_ip, server_port))
        self.socket.listen(1)

    def server_handshake(self):
        (connection, addr) = self.socket.accept()
        self.connection = connection
        handshake = self.connection.recv(1)

        if handshake == b'\00':
            self.logger.log_info('Successfully received client Handshake; sending handshake back')
            connection.sendall(b'\00')
        else:
            self.logger.log_error('Message received not client handshake')

    def start(self, server_ip, server_port):
        self.logger.log_info('Starting server')
        self.listen(server_ip, server_port)
        self.server_handshake()
        self.logger.log_info('Successfully started server and handshake')

    def get_client_data(self):
        '''parses message and returns the client data (effectively message['data'])'''

        # collected_message = False
        self.logger.log_info('Waiting for client...')
        while True:
            bs = self.connection.recv(8)
            (length,) = unpack('>Q', bs)
            data = b''
            while len(data) < length:
                to_read = length - len(data)
                data += self.connection.recv(
                    self.socket_buffer_read_size if to_read > self.socket_buffer_read_size else to_read)

            # send our 0 ack
            self.connection.sendall(b'\00')

            return pickle.loads(data)

    def parse_message(self, message):
        '''logs the latency (message['latency']) and returns the data (message['data')'''
        timestamp = message['timestamp']
        data = message['data']

        latency = time.time() - timestamp

        if latency <= 1e-2:
            self.logger.log_debug(f'Message sent at timestamp {timestamp} and received with latency {latency}')

        self.logger.log_info(f'Received data with latency {round(latency, 4)}')
        self.stats_logger.push_log({'latency' : round(latency, 4)}, append=False)

        return data

    def process_data(self, client_data):
        '''processes the message using one of the detection models'''
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
            model_outputs = self.server_model([torch.from_numpy(decoded_frame).float().to(self.server_device)])[
                0]  # first frame
        else:
            raise NotImplementedError('No other compression method exists.')

        self.logger.log_debug("Generated new BB with model (online).")
        cpu_model_outputs = move_data_dict_to_device(model_outputs, 'cpu')
        if 'masks' in cpu_model_outputs:
            del cpu_model_outputs['masks']
        return cpu_model_outputs

    def start_server_loop(self):
        '''main loop'''
        try:
            iteration_num = 0
            # effectively time waiting for client
            time_since_processed_lass_message = time.time()
            while True:
                client_data = self.get_client_data()
                if PARAMS['FILE_TRANSFER'] and 'list' in str(type(client_data)):
                    # send data over
                    self.logger.log_info(f'Received fname list of {len(client_data)} files')
                    byte_files = []
                    for fname in client_data:
                        with open(f'{PARAMS["DATA_DIR"]}/{fname}', 'rb') as f:
                            byte_files.append(f.read())

                    self.logger.log_debug(f'Sent response of byte files: size {sum(sys.getsizeof(x) for x in byte_files)}')
                    self.send_response(byte_files)

                else:
                    data = self.parse_message(client_data) # data  .shape
                    self.logger.log_debug(f'Finished getting client data.')

                    curr_time = time.time()
                    self.stats_logger.push_log({'client_time' : curr_time - time_since_processed_lass_message})
                    response = self.process_data(data)
                    process_time = round(time.time() - curr_time, 4)

                    response_size = get_tensor_size(response)

                    self.logger.log_info(f'Sending response of size {response_size}.')

                    self.send_response(response)

                    self.stats_logger.push_log({'processing_time' : process_time, 'iteration' : iteration_num,
                                                'response_size' : response_size}, append=True)
                    iteration_num += 1

                time_since_processed_lass_message = time.time()

        except Exception as ex:
            traceback.print_exc()
            self.logger.log_error(ex)
            self.close()

        finally:
            self.logger.log_info('Server Loop Ended')

    def send_response(self, data):
        data = pickle.dumps(data)
        length = pack('>Q', len(data))

        self.connection.sendall(length)
        self.connection.sendall(data)

        ack = self.connection.recv(1)
        self.logger.log_debug(f'Received the ack {ack} from the response.')

    def close(self):
        self.stats_logger.flush()
        if not self.server_connect:
            return
        self.connection.shutdown(SHUT_WR)
        self.connection.close()
        self.socket.close()
        self.socket = None

#main functionality for testing/debugging
if __name__ == '__main__':
    server = Server()

    server.start(PARAMS['HOST'], PARAMS['PORT'])
    server.start_server_loop()
    server.close()
