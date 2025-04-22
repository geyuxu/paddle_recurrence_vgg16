import paddle.vision.transforms as T
from paddle.vision.datasets import Cifar10 #, ImageNet

def get_transforms(input_size):
    return T.Compose([
        T.Resize(input_size),
        T.Transpose(),
        T.Normalize(mean=[127.5]*3, std=[127.5]*3),
    ])

def get_dataset(name, mode, transform):
    if name == 'cifar10':
        return Cifar10(mode=mode, transform=transform)
    # elif name == 'imagenet':
    #     return ImageNet(mode=mode, transform=transform)
    elif name == 'custom':
        raise NotImplementedError("自定义数据集未实现")
    else:
        raise ValueError(f"Unsupported dataset: {name}")
