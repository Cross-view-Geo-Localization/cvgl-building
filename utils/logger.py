import os
import sys
import random
import errno
import torch
import numpy as np
import logging
# from clearml import Task

# class ExperimentLogger:
#     def __init__(self, project_name, task_name, config_dict):
#         # Setup Python logging
#         logging.basicConfig(
#             level=logging.INFO,
#             format='%(asctime)s - %(levelname)s - %(message)s'
#         )
#         self.console = logging.getLogger(__name__)

#         # Setup ClearML logging
#         self.task = Task.init(project_name=project_name, task_name=task_name)
#         self.task.connect(config_dict, name="Experiment Config")

#         self.clearml_logger = self.task.get_logger()
#         self.console.info(f"Started experiment: {task_name} in project: {project_name}")

#     def info(self, message):
#         self.console.info(message)

#     def log_scalar(self, title, series, value, step):
#         """
#         Logs a scalar value (like loss or Recall@K) to generate plots.
        
#         Args:
#             title: The main graph title (e.g., 'Training Loss')
#             series: The specific line on the graph (e.g., 'Triplet Loss')
#             value: The numeric value
#             step: The current epoch or batch iteration
#         """
#         self.clearml_logger.report_scalar(
#             title=title, 
#             series=series, 
#             value=value, 
#             iteration=step
#         )

#     def close(self):
#         self.task.close()
#         self.console.info("Experiment finished and logger closed.")


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def isatty(self):
        # Check if the underlying console/stdout is a terminal
        return hasattr(self.console, 'isatty') and self.console.isatty()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def setup_system(seed, cudnn_benchmark=True, cudnn_deterministic=True) -> None:
    '''
    Set seeds for for reproducible training
    '''
    # python
    random.seed(seed)
    
    # numpy
    np.random.seed(seed)
    
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn_benchmark_enabled = cudnn_benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic
      
        
def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise