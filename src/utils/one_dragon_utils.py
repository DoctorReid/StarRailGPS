import os
import random
import shutil
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml
from cv2.typing import MatLike

from basic import Rect, Point
from utils import os_utils, cv2_utils, data_set_utils
from utils.log_utils import log


def get_original_one_dragon_dir():
    """
    原一条龙项目的文件夹位置
    :return:
    """
    return os_utils.get_env('ONE_DRAGON_DIR')


def get_data_dir():
    """
    当前项目一条龙数据的位置
    :return:
    """
    return os_utils.get_path_under_work_dir('one_dragon_data')


def get_cal_pos_dir():
    """
    当前项目一条龙坐标数据的位置
    :return:
    """
    dir = os.path.join(get_data_dir(), 'cal_pos')
    if not os.path.exists(dir):
        os.mkdir(dir)
    return dir


def get_region_map_dir():
    """
    当前项目一条龙大地图的位置
    :return:
    """
    dir = os.path.join(get_data_dir(), 'region_map')
    if not os.path.exists(dir):
        os.mkdir(dir)
    return dir


def sync_original_cal_pos(target_region: Optional[str] = None, overwrite: bool = False):
    """
    从一条龙项目中同步原始坐标数据过来
    :param target_region: 只同步特定的区域
    :param overwrite: 是否覆盖写入 按区域覆盖
    :return:
    """
    od_cal_pos_dir = os.path.join(get_original_one_dragon_dir(), '.debug', 'cal_pos')

    if not os.path.exists(od_cal_pos_dir):
        log.info('无可同步内容')
        return

    curr_cal_pos_dir = get_cal_pos_dir()

    sync_cnt: int = 0
    region_dir_list = os.listdir(od_cal_pos_dir)
    for region_id in region_dir_list:
        if target_region is not None and region_id != target_region:
            continue
        original_region_dir = os.path.join(od_cal_pos_dir, region_id)
        if not os.path.isdir(original_region_dir):
            continue

        current_region_dir = os.path.join(curr_cal_pos_dir, region_id)
        if overwrite and os.path.exists(current_region_dir):
            shutil.rmtree(current_region_dir)

        if not os.path.exists(current_region_dir):
            os.mkdir(current_region_dir)

        case_dir_list = os.listdir(original_region_dir)
        for case in case_dir_list:
            od_case_dir = os.path.join(original_region_dir, case)
            if not os.path.isdir(od_case_dir):
                continue

            idx: int = 1
            while True:
                curr_case_dir = os.path.join(current_region_dir, '%03d' % idx)
                if not os.path.exists(curr_case_dir):
                    shutil.copytree(od_case_dir, curr_case_dir)
                    sync_cnt += 1
                    break
                idx += 1

    log.info('总共同步 %d 个样例', sync_cnt)


def sync_original_region_map(target_region: Optional[str] = None):
    """
    从一条龙项目中同步原始大地图图片过来
    :param target_region: 只同步特定的区域
    :return:
    """
    sync: int = 0
    od_region_map_dir = os.path.join(get_original_one_dragon_dir(), 'images', 'map')
    curr_region_map_dir = get_region_map_dir()

    planet_list = os.listdir(od_region_map_dir)
    for planet in planet_list:
        planet_dir = os.path.join(od_region_map_dir, planet)
        if not os.path.isdir(planet_dir):
            continue
        region_list = os.listdir(planet_dir)
        for region in region_list:
            region_dir = os.path.join(planet_dir, region)
            if not os.path.isdir(region_dir):
                continue

            file_list = ['raw.png', 'origin.png']
            region_id = '%s_%s' % (planet, region)
            if target_region is not None and region_id != target_region:
                continue

            for file in file_list:
                file_path = os.path.join(region_dir, file)
                if not os.path.exists(file_path) or not os.path.isfile(file_path):
                    continue

                curr_file_path = os.path.join(curr_region_map_dir, '%s.png' % region_id)
                shutil.copy2(file_path, curr_file_path)
                sync += 1
                log.info('同步大地图图片 %s', region_id)
                break  # 复制其中一张图即可

    log.info('总共同步 %d 张图片', sync)


def sync_to_data_set(version: int = 0, target_region: Optional[str] = None, overwrite: bool = False):
    sync: int = 0
    map_dir = get_region_map_dir()
    region_list = os.listdir(map_dir)
    data_set_dir = data_set_utils.get_data_set_dir(version)
    for region_map_path in region_list:
        if not region_map_path.endswith('.png'):
            continue
        region_id = region_map_path[:-4]
        if target_region is not None and region_id != target_region:
            continue

        cal_pos_dir = os.path.join(get_cal_pos_dir(), region_id)
        if not os.path.exists(cal_pos_dir):
            log.error('原始坐标数据文件夹不存在 %s', region_id)
            continue

        region_data_dir = os.path.join(data_set_dir, region_id)
        if overwrite and os.path.exists(region_data_dir):
            shutil.rmtree(region_data_dir)

        if not os.path.exists(region_data_dir):
            os.mkdir(region_data_dir)

        region_map = cv2_utils.read_image(os.path.join(map_dir, region_map_path))

        od_case_list = os.listdir(cal_pos_dir)
        for od_case in od_case_list:
            od_case_dir = os.path.join(cal_pos_dir, od_case)

            idx = 1
            while True:
                data_case_dir = os.path.join(region_data_dir, '%03d' % idx)
                if not os.path.exists(data_case_dir):
                    os.mkdir(data_case_dir)
                    break
                idx += 1

            od_data = None
            od_case_data_path = os.path.join(od_case_dir, 'pos.yml')
            try:
                with open(od_case_data_path, 'r', encoding='utf-8') as file:
                    od_data = yaml.safe_load(file)
            except Exception:
                log.error('yml文件读取失败 %s %s', region_id, od_case)

            if od_data is None:
                log.error('无法读取坐标数据 %s %s', region_id, od_case)
                continue

            od_mm_path = os.path.join(od_case_dir, 'mm.png')
            if not os.path.exists(od_mm_path):
                log.error('缺少小地图截图 %s %s', region_id, od_case)
                continue

            case_match_rect = Rect(od_data['x'], od_data['y'],
                                   od_data['x'] + od_data['w'], od_data['y'] + od_data['h'])

            lm, bias = random_crop_region_map(region_map, case_match_rect.center)

            od_data['od_x'] = od_data['x']
            od_data['x'] = od_data['od_x'] + bias.x
            od_data['od_y'] = od_data['y']
            od_data['y'] = od_data['od_y'] + bias.y

            data_mm_path = os.path.join(data_case_dir, 'mm.png')
            shutil.copy2(od_mm_path, data_mm_path)

            data_lm_path = os.path.join(data_case_dir, 'lm.png')
            cv2.imwrite(data_lm_path, lm)

            data_pos_path = os.path.join(data_case_dir, 'pos.yml')
            with open(data_pos_path, 'w', encoding='utf-8') as file:
                yaml.dump(od_data, file)

            sync += 1
            log.info('同步完成 %s %s', region_id, od_case)

    log.info('总共同步 %d 份样例', sync)


def random_crop_region_map(region_map: MatLike, pos: Point) -> Tuple[MatLike, Point]:
    """
    :param region_map: 大地图
    :param pos: 原匹配结果的中心点
    :return: 裁剪得到的大地图 和 中心坐标的偏移量
    """
    # 人物跑动
    run_speed: int = 30
    run_seconds: int = 2
    run_dis = run_speed * run_seconds
    # 随机偏移一段距离 代表人物上一次所在的坐标 也是大地图上需要裁剪的中心
    bias = Point(random.randint(-run_dis, run_dis), random.randint(-run_dis, run_dis))
    center = pos + bias
    # 裁剪的宽度 = 人物跑动距离 + 小地图半径(95) + 容错(10)
    crop_r = run_dis + 95 + 10
    crop_d = crop_r * 2
    # 裁剪的框
    crop_rect = Rect(center.x - crop_r, center.y - crop_r,
                     center.x + crop_r, center.y + crop_r)
    # 裁剪后的大地图 由于超出边界问题 这个图可能是不完整的
    lm, real_crop_rect = cv2_utils.crop_image(region_map, crop_rect)

    crop_map = np.zeros((crop_d, crop_d, 3))
    sx = 0 if crop_rect.x1 >= 0 else -crop_rect.x1
    sy = 0 if crop_rect.y1 >= 0 else -crop_rect.y1
    ex = sx + real_crop_rect.width
    ey = sy + real_crop_rect.height
    # 裁剪的图片
    crop_map[sy:ey, sx:ex] = lm[:,:]
    # 边缘部分使用背景色填充
    bg_color = (200, 200, 200)
    if sx > 0:
        crop_map[:, 0:sx] = bg_color
    if sy > 0:
        crop_map[0:sy, :] = bg_color
    if ex - sx < crop_rect.width:
        crop_map[:, ex:] = bg_color
    if ey - sy < crop_rect.height:
        crop_map[ey:, :] = bg_color

    # 真正的偏移量
    real_bias = Point(0, 0) - real_crop_rect.left_top

    return crop_map, real_bias
