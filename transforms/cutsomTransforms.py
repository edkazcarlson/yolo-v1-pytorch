from PIL import Image
from torchvision import transforms
from .rgb2xyl import rgb2xylTransform
import numpy as np
import torch
import torch.nn.functional as F

# tuple 224, 224
t224 = (224, 224)
camvidSize = (360, 480)

def getResize(dataSet):
    size = t224
    if dataSet == 'camvid':
        size = camvidSize
    return size


def xylTransform(normalized):
    [
        rgb2xylTransform(normalize=normalized),
]
