import os

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.ops import nms


def get_new_img_size(width, height, img_min_side=600):
    if width <= height:
        f = float(img_min_side) / width
        resize_height = int(f * height)
        resize_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resize_height = int(img_min_side)
        resize_width = int(f * width)

    return resize_width, resize_height

def bbox2loc(src_bbox, dst_bbox):
    width = src_bbox[:,2] - src_bbox[:,0]
    height = src_bbox[:,3] - src_bbox[:,1]
    ctr_x = src_bbox[:,0] + width/2
    ctr_y = src_bbox[:,1] + height/2

    base_width = dst_bbox[:,2] - dst_bbox[:,0]
    base_height = dst_bbox[:,3] - dst_bbox[:,1]
    base_ctr_x = dst_bbox[:,0] + base_width/2
    base_ctr_y = dst_bbox[:,1] + base_height/2

    #eps初始为一个极小的正值，保证height和width为正

    eps = np.finfo(height.dtype).eps
    width = np.maximum(width, eps)
    height = np.maximum(height, eps)

    dx = (base_ctr_x - ctr_x) / width
    dy = (base_ctr_y - ctr_y) / height
    dw = np.log(base_width / width)
    dh = np.log(base_height / height)

    loc = np.vstack((dx,dy,dw,dh)).transpose()
    return loc

def loc2bbox(src_bbox, loc):
    if src_bbox.size()[0] == 0:
        return torch.zeros((0,4), dtype=loc.dtype)

    src_width = torch.unsqueeze(src_bbox[:,2] - src_bbox[:,0],-1)
    src_height = torch.unsqueeze(src_bbox[:,3] - src_bbox[:,1],-1)
    src_ctrx = torch.unsqueeze(src_bbox[:,0],-1) + src_width/2
    src_ctry = torch.unsqueeze(src_bbox[:,1],-1) + src_height/2

    dx = loc[:,0::4]
    dy = loc[:,1::4]
    dw = loc[:,2::4]
    dh = loc[:,3::4]

    ctr_x = dx * src_width + src_ctrx
    ctr_y = dy * src_height + src_ctry
    w = torch.exp(dw) * src_width
    h = torch.exp(dh) * src_height
    dst_bbox = torch.zeros_like(loc)
    dst_bbox[:,0::4] = ctr_x - 0.5 * w
    dst_bbox[:,1::4] = ctr_y - 0.5 * h
    dst_bbox[:,2::4] = ctr_x + 0.5 * w
    dst_bbox[:,3::4] = ctr_y + 0.5 * h

    return dst_bbox