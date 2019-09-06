import os
import datetime
import logging

class prjPaths:
    def __init__(self):
        self.SRC_DIR = os.path.abspath(os.path.curdir)
        self.ROOT_MOD_DIR = '/'.join(self.SRC_DIR.split('/'))
        self.ROOT_DATA_DIR = os.path.join(self.ROOT_MOD_DIR,'data')
        self.LIB_DIR = os.path.join(self.ROOT_MOD_DIR,'lib')
        self.CHECKPOINT_DIR = os.path.join(self.LIB_DIR,'chkpts')
        self.SUMMARY_DIR = os.path.join(self.LIB_DIR,'summaries')
        self.LOGS_DIR = os.path.join(self.LIB_DIR,'logs')


        pth_exists_else_mk = lambda path:os.mkdir(path) if not os.path.exists(path) else None

        pth_exists_else_mk(self.ROOT_DATA_DIR)
        pth_exists_else_mk(self.LIB_DIR)
        pth_exists_else_mk(self.CHECKPOINT_DIR)
        pth_exists_else_mk(self.SUMMARY_DIR)
        pth_exists_else_mk(self.LOGS_DIR)

def get_logger(paths):
    currentTime = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    logFileName = os.path.join(paths.LOGS_DIR,'HAN_TextClassification_{}.log'.format(currentTime))
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctim)s:%(name)s:%(message)s')

    fileHandler = logging.FileHandler(logFileName)
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)

    logger.addHandler(fileHandler)

    return logger

class Configer:
    def __init__(self):
        self.dataset = 'imdb'
        self.run_type = 'train'
        self.emebdding_dim = 100
        self.batch_size = 2
        self.num_epochs = 25
        self.evaluate_every = 100
        self.log_summaries_every = 30
        self.checkpoint_every = 100
        self.num_checkpoints = 5
        self.max_grad_norm = 5.0
        self.dropout_keep_proba = 0.5
        self.learning_rate = 1e-3
        self.per_process_gpu_memory_fraction = 0.9
