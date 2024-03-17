import logging
import os
from logging.handlers import TimedRotatingFileHandler

from utils import os_utils


def get_logger():
    logger = logging.getLogger('StarRailGPS')
    logger.handlers.clear()

    formatter = logging.Formatter('[%(asctime)s] [%(filename)s %(lineno)d] [%(levelname)s]: %(message)s', '%H:%M:%S')

    log_file_path = os.path.join(os_utils.get_path_under_work_dir('.log'), 'log.txt')
    archive_handler = TimedRotatingFileHandler(log_file_path, when='midnight', interval=1, backupCount=3, encoding='utf-8')
    archive_handler.setFormatter(formatter)
    logger.addHandler(archive_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


log = get_logger()
