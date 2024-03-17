import os
from typing import List, Tuple

import cv2
from cv2.typing import MatLike
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader

from utils import data_set_utils, cv2_utils, yaml_utils


class StarRailGPSDataItemV1:

    def __init__(self, version: int, region_id: str, case_id: str):
        data_dir = data_set_utils.get_data_set_dir(version)
        region_dir = os.path.join(data_dir, region_id)
        case_dir = os.path.join(region_dir, case_id)

        self.mm: MatLike = cv2_utils.read_image(os.path.join(case_dir, 'mm.png'))
        self.lm: MatLike = cv2_utils.read_image(os.path.join(case_dir, 'lm.png'))
        self.data: dict = yaml_utils.read_file(os.path.join(case_dir, 'pos.yml'))

        # 转化成resnet用的224*224
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 大部分预训练的模型都是RGB格式
        self.mm_tensor = transform(cv2.cvtColor(self.mm, cv2.COLOR_BGR2RGB))
        self.lm_tensor = transform(cv2.cvtColor(self.lm, cv2.COLOR_BGR2RGB))

        # 最终结果
        self.ans = torch.tensor([self.data['x'], self.data['y'],
                                 self.data['x'] + self.data['w'],
                                 self.data['y'] + self.data['h']
                                 ], dtype=torch.uint8)


class StarRailGPSDatasetV1(Dataset):

    def __init__(self, version: int):
        self.data_list: List[StarRailGPSDataItemV1] = []
        data_dir = data_set_utils.get_data_set_dir(version)
        for region_id in os.listdir(data_dir):
            region_dir = os.path.join(data_dir, region_id)
            if not os.path.isdir(region_dir):
                continue

            for case_id in os.listdir(region_dir):
                case_dir = os.path.join(region_dir, case_id)
                if not os.path.isdir(case_dir):
                    continue

                self.data_list.append(StarRailGPSDataItemV1(version, region_id, case_id))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


class StarRailGPSV1(nn.Module):

    def __init__(self):
        super(StarRailGPSV1, self).__init__()
        # 为两张图片使用两个不同的ResNet预训练模型
        self.mm_resnet = resnet18(pretrained=False)
        self.lm_resnet = resnet18(pretrained=False)
        # 去除全连接层
        self.mm_resnet.fc = nn.Identity()
        self.lm_resnet.fc = nn.Identity()

        # 特征融合和定位网络
        self.fc = nn.Sequential(
            nn.Linear(2048 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 4)  # 假设输出格式为 [x1, y1, x2, y2]
        )

    def forward(self, img1, img2):
        # 提取特征
        feat1 = self.resnet1(img1)
        feat2 = self.resnet2(img2)
        # 特征融合
        features = torch.cat((feat1, feat2), dim=1)
        # 定位
        coords = self.fc(features)
        return coords


def get_data(version: int) -> Tuple[Dataset, DataLoader]:
    """
    加载一个版本的数据
    :param version:
    :return:
    """
    dataset = StarRailGPSDatasetV1(version)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataset, dataloader

def train(model: StarRailGPSV1, dataloader: DataLoader):
    pass
