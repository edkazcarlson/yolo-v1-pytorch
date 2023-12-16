import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# https://docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html#color_convert_rgb_hls
# Make a chroma cone by doing abs(lightness - .5) * s for the saturation
# https://en.wikipedia.org/wiki/HSL_and_HSV#/media/File:HSL_color_solid_dblcone_chroma_gray.png
# a and b are the x y coordinates on the cross section
def rgb2xyl(x):
    if (type(x) == type(torch.tensor([]))):
        x = x.transpose(0,1).transpose(1,2) #todo figure out what I meant for this
        x = np.float32(np.asarray(x))
    elif type(x) == type(Image.Image()):
        x = np.float32(np.asarray(x))
        x /= 255
        assert np.min(x) >= 0 
        assert np.max(x) <= 1
    else: 
        print(f'found {type(x)} wanted {type(Image.Image())}')
        exit()
        
    assert np.max(x) <= 1
    assert np.min(x) >= 0

    x = cv2.cvtColor(x, cv2.COLOR_RGB2HLS)
    x = torch.tensor(x, dtype= torch.float)
    x = x.transpose(2,1)
    x = x.transpose(1,0)
    # h is in range 0-360, s and v are 0-1

    x = x.float()
    h = x[0]
    l = x[1]
    s = x[2]
    # print(f's[0,0]: {s[0,0]}')
    # print(s.shape)
    assert torch.max(h) <= 360
    assert torch.min(h) >= 0
    assert torch.max(s) <= 1.0001, f'torch.max: {torch.max(s)}'
    assert torch.min(s) >= 0
    assert torch.max(l) <= 1.0001
    assert torch.min(l) >= 0

    # s = 1 - 2 * torch.abs(l - .5) #no cone mode
    h = torch.pi * 2 * (h / 360)
    x = torch.cos(h)
    y = torch.sin(h)
    x = s * x
    y = s * y
    xyl = torch.stack((x,y,l))
    return xyl

class rgb2xylTransform(object):
    """Changes an image from bgr to xyl.

    Args: normalize: normalizes
    """
    def __init__(self, normalize: bool):
        self.normalize = normalize
        self.xylNormalize = transforms.Normalize([0.04220286750428868,0.0524272059900699,0.4725254927987115],
                                            [0.24847656724988818,0.1885309268079895,0.23849825834988292])
    def __call__(self, x):
        x = rgb2xyl(x)
        if self.normalize:
            return self.xylNormalize(x)
        else:
            return x
    
# ----------
# min
# -0.9999998807907104
# max
# 1.0
# mean
# 0.011003296055454708
# std
# 0.47058728456633464
# ----------
# min
# -0.9999999403953552
# max
# 1.0
# mean
# -0.004574688462443411
# std
# 0.4612628221288517
# ----------
# min
# 0.0
# max
# 1.0
# mean
# 0.4725254927987115
# std
# 0.23849825834988292
