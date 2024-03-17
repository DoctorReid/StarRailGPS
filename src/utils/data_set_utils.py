import os
import cv2
import random

import yaml

from utils import os_utils, cv2_utils
from utils.log_utils import log


def get_data_set_dir(version: int = 0) -> str:
    """
    获取数据集的目录
    """
    return os_utils.get_path_under_work_dir('data_set', '%02d' % version)


def show_data_set_case(region_id: str, case: int = 0, version: int = 0):
    """
    展示某个数据
    :param region_id: 区域
    :param case: 样例编号 不传入时随机一个
    :param version: 数据版本号
    """
    data_set_dir = get_data_set_dir(version=version)
    region_dir = os.path.join(data_set_dir, region_id)
    if not os.path.exists(region_dir):
        log.error('区域文件夹不存在 %02d %s', version, region_id)
        return

    if case == 0:
        case_id_list = os.listdir(region_dir)
        case_id = random.choice(case_id_list)
        log.info('随机使用样例 %s', case_id)
    else:
        case_id = '%03d' % case

    case_dir = os.path.join(region_dir, case_id)
    lm_path = os.path.join(case_dir, 'lm.png')
    if not os.path.exists(lm_path):
        log.error('缺少大地图图片 %02d %s %s', version, region_id, case_id)
        return
    mm_path = os.path.join(case_dir, 'mm.png')
    if not os.path.exists(mm_path):
        log.error('缺少小地图图片 %02d %s %s', version, region_id, case_id)
        return
    pos_path = os.path.join(case_dir, 'pos.yml')
    if not os.path.exists(pos_path):
        log.error('缺少坐标数据 %02d %s %s', version, region_id, case_id)
        return

    lm = cv2_utils.read_image(lm_path)
    mm = cv2_utils.read_image(mm_path)
    with open(pos_path, 'r', encoding='utf-8') as file:
        pos_data = yaml.safe_load(file)

    to_show_mm = cv2.resize(mm, (pos_data['w'], pos_data['h']))
    cv2_utils.show_overlap(lm, to_show_mm, pos_data['x'], pos_data['y'])
