import pytorch

from torch.utils.data import Dataset
from utils import data_set_utils

class SimpleDataset(Dataset):

    def __init__(self, version: int):
        data_set_utils.