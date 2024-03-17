import os
from typing import Optional, Tuple

import cv2
from cv2.typing import MatLike

from basic import Rect


def read_image(file_path: str) -> Optional[MatLike]:
    """
    读取图片
    :param file_path: 图片路径
    :return:
    """
    if not os.path.exists(file_path):
        return None
    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    return image


def crop_image(img, rect: Optional[Rect] = None, copy: bool = False) -> Tuple[MatLike, Optional[Rect]]:
    """
    裁剪图片 裁剪区域可能超出图片范围
    :param img: 原图
    :param rect: 裁剪区域 (x1, y1, x2, y2)
    :param copy: 是否复制新图
    :return: 裁剪后图片 和 实际的裁剪区域
    """
    if rect is None:
        return (img.copy() if copy else img), rect

    x1, y1, x2, y2 = rect.x1, rect.y1, rect.x2, rect.y2
    if x1 < 0:
        x1 = 0
    if x2 > img.shape[1]:
        x2 = img.shape[1]
    if y1 < 0:
        y1 = 0
    if y2 > img.shape[0]:
        y2 = img.shape[0]

    x1, y1 = int(x1), int(y1)
    x2, y2 = int(x2), int(y2)
    crop = img[y1: y2, x1: x2]
    return (crop.copy() if copy else crop), Rect(x1, y1, x2, y2)


def crop_image_only(img, rect: Optional[Rect] = None, copy: bool = False) -> MatLike:
    """
    裁剪图片 裁剪区域可能超出图片范围
    :param img: 原图
    :param rect: 裁剪区域 (x1, y1, x2, y2)
    :param copy: 是否复制新图
    :return: 只返回裁剪后图片
    """
    return crop_image(img, rect=rect, copy=copy)[0]


def source_overlap_template(source: MatLike, template: MatLike, x: int, y: int):
    """
    复制一张新图 在原图上覆盖模板图
    :param source: 原图
    :param template: 模板图 缩放后
    :param x: 偏移量
    :param y: 偏移量
    :return:
    """
    to_overlap_source = source.copy()

    rect1, rect2 = get_overlap_rect(source, template, x, y)
    sx_start, sy_start, sx_end, sy_end = rect1
    tx_start, ty_start, tx_end, ty_end = rect2

    # 将覆盖图像放置到底图的指定位置
    to_overlap_source[sy_start:sy_end, sx_start:sx_end] = template[ty_start:ty_end, tx_start:tx_end]

    return to_overlap_source


def get_overlap_rect(source, template, x, y):
    """
    根据模板图在原图上的偏移量 计算出覆盖区域
    :param source: 原图
    :param template: 模板图 缩放后
    :param x: 偏移量
    :param y: 偏移量
    :return:
    """
    # 获取要覆盖图像的宽度和高度
    overlay_height, overlay_width = template.shape[:2]

    # 覆盖图在原图上的坐标
    sx_start = int(x)
    sy_start = int(y)
    sx_end = sx_start + overlay_width
    sy_end = sy_start + overlay_height

    # 覆盖图要用的坐标
    tx_start = 0
    ty_start = 0
    tx_end = overlay_width
    ty_end = overlay_height

    # 覆盖图缩放后可以超出了原图的范围
    if sx_start < 0:
        tx_start -= sx_start
        sx_start -= sx_start
    if sx_end > source.shape[1]:
        tx_end -= sx_end - source.shape[1]
        sx_end -= sx_end - source.shape[1]

    if sy_start < 0:
        ty_start -= sy_start
        sy_start -= sy_start
    if sy_end > source.shape[0]:
        ty_end -= sy_end - source.shape[0]
        sy_end -= sy_end - source.shape[0]

    return (sx_start, sy_start, sx_end, sy_end), (tx_start, ty_start, tx_end, ty_end)
