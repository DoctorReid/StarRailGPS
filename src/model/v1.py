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

from utils import data_set_utils, cv2_utils, yaml_utils, pytorch_utils
from utils.log_utils import log


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
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 大部分预训练的模型都是RGB格式
        self.mm_tensor = transform(cv2.cvtColor(self.mm, cv2.COLOR_BGR2RGB))
        self.lm_tensor = transform(cv2.cvtColor(self.lm, cv2.COLOR_BGR2RGB))

        # 最终结果
        self.ans = torch.tensor([self.data['x'], self.data['y'],
                                 self.data['x'] + self.data['w'],
                                 self.data['y'] + self.data['h']
                                 ], dtype=torch.float)


class StarRailGPSDatasetV1(Dataset):

    def __init__(self, version: int, max_size: int = -1):
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
            if max_size != -1 and len(self.data_list) > max_size:
                break
        log.info(f'Dataset加载完毕 总共 {len(self.data_list)}')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item: StarRailGPSDataItemV1 = self.data_list[idx]
        return item.mm_tensor, item.lm_tensor, item.ans


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
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # 假设输出格式为 [x1, y1, x2, y2]
        )

    def forward(self, mm, lm):
        # 提取特征
        feat1 = self.mm_resnet(mm)
        feat2 = self.lm_resnet(lm)
        # 特征融合
        features = torch.cat((feat1, feat2), dim=1)
        # 定位
        coords = self.fc(features)
        return coords


def get_data(version: int, batch_size: int = 32, max_size: int = -1) -> Tuple[Dataset, DataLoader]:
    """
    加载一个版本的数据
    :param version: 版本号
    :param batch_size: 批次数量
    :param max_size: 最大数量
    :return:
    """
    dataset = StarRailGPSDatasetV1(version, max_size=max_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, dataloader


def train(model: StarRailGPSV1, train_loader: DataLoader, test_loader: DataLoader,
          criterion, optimizer,
          device: str = 'cpu',
          num_epochs: int = 1000):
    """
    对一个模型进行训练
    :param model: 模型
    :param train_loader:
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param device: cpu/cuda
    :param num_epochs: 迭代次数
    :return:
    """
    for epoch in range(num_epochs):
        # 初始化预测值和真实值的列表
        all_predictions = []
        all_targets = []
        for batch_idx, (mm, lm, ans) in enumerate(train_loader):
            mm_device = mm.to(device)
            lm_device = lm.to(device)
            ans_device = ans.to(device)
            # 前向传播
            output = model(mm_device, lm_device)
            # 计算损失
            loss = criterion(output, ans_device)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            # 更新参数
            optimizer.step()

            # 将预测值和真实值添加到列表中
            all_predictions.extend(output.tolist())
            all_targets.extend(ans_device.tolist())

            if batch_idx % 10 == 0:
                log.info(f'Epoch {epoch} Batch {batch_idx} Done. Loss: {loss.item()}')

        if epoch % 10 == 0:
            epoch_loss = criterion(torch.tensor(all_predictions), torch.tensor(all_targets))
            test_loss = test(model, test_loader, criterion)
            log.info(f"Epoch {epoch}: Train loss {epoch_loss.item()}. Test loss {test_loss}")


def test(model: StarRailGPSV1, loader: DataLoader, criterion):
    """
    在训练集上测试模型
    :param model:
    :param loader:
    :param criterion:
    :return:
    """
    with torch.no_grad():
        # 初始化预测值和真实值的列表
        all_predictions = []
        all_targets = []
        for batch_idx, (mm, lm, ans) in enumerate(loader):
            mm_device = mm.to(device)
            lm_device = lm.to(device)
            ans_device = ans.to(device)
            # 前向传播
            output = model(mm_device, lm_device)

            # 将预测值和真实值添加到列表中
            all_predictions.extend(output.tolist())
            all_targets.extend(ans_device.tolist())

        # 计算当前epoch的整体误差
        return criterion(torch.tensor(all_predictions), torch.tensor(all_targets))


if __name__ == '__main__':
    device = pytorch_utils.get_default_device()
    model = StarRailGPSV1()
    train_dataset, train_loader = get_data(version=1, max_size=50)
    test_data_set, test_loader = get_data(version=1, max_size=50)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.to(device)
    criterion.to(device)

    train(model, train_loader, test_loader, criterion, optimizer, num_epochs=20)
