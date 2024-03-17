import torch


def get_default_device() -> str:
    """
    根据硬件自动获取device类型
    :return:
    """
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )