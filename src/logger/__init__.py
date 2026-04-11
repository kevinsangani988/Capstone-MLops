import logging
import os
from datetime import datetime
import time

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
dir_name = 'logging_details'

os.makedirs(dir_name, exist_ok=True)

file_name = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

file_path = os.path.join(dir_name, file_name)

def create_logger():

    logger = logging.getLogger('Base logger')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s -- %(levelname)s -- %(message)s -- %(name)s')

    filehandler = logging.FileHandler(file_path)
    filehandler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(filehandler)
    logger.addHandler(console_handler)
    return logger
logger = create_logger()