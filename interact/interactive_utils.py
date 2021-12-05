# Modifed from https://github.com/seoungwugoh/ivs-demo

import numpy as np
import os
import copy
import cv2
import glob

import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from PIL import Image, ImageDraw, ImageFont

import torch
from torchvision import models
from dataset.range_transform import im_normalization


def images_to_torch(frames, device):
    frames = torch.from_numpy(frames.transpose(0, 3, 1, 2)).float().unsqueeze(0)/255
    b, t, c, h, w = frames.shape
    for ti in range(t):
        frames[0, ti] = im_normalization(frames[0, ti])
    return frames.to(device)

def load_images(path, min_side=None):
    fnames = sorted(glob.glob(os.path.join(path, '*.jpg')))
    if len(fnames) == 0:
        fnames = sorted(glob.glob(os.path.join(path, '*.png')))
    frame_list = []
    for i, fname in enumerate(fnames):
        if min_side:
            image = Image.open(fname).convert('RGB')
            w, h = image.size
            new_w = (w*min_side//min(w, h))
            new_h = (h*min_side//min(w, h))
            frame_list.append(np.array(image.resize((new_w, new_h), Image.BICUBIC), dtype=np.uint8))
        else:
            frame_list.append(np.array(Image.open(fname).convert('RGB'), dtype=np.uint8))
    frames = np.stack(frame_list, axis=0)
    return frames

def load_masks(path, min_side=None):
    fnames = sorted(glob.glob(os.path.join(path, '*.png')))
    frame_list = []

    first_frame = np.array(Image.open(fnames[0]))
    binary_mask = (first_frame.max() == 255)

    for i, fname in enumerate(fnames):
        if min_side:
            image = Image.open(fname)
            w, h = image.size
            new_w = (w*min_side//min(w, h))
            new_h = (h*min_side//min(w, h))
            frame_list.append(np.array(image.resize((new_w, new_h), Image.NEAREST), dtype=np.uint8))
        else:
            frame_list.append(np.array(Image.open(fname), dtype=np.uint8))

    frames = np.stack(frame_list, axis=0)
    if binary_mask:
        frames = (frames > 128).astype(np.uint8)
    return frames

def load_video(path, min_side=None):
    frame_list = []
    cap = cv2.VideoCapture(path)
    while(cap.isOpened()):
        _, frame = cap.read()
        if frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if min_side:
            h, w = frame.shape[:2]
            new_w = (w*min_side//min(w, h))
            new_h = (h*min_side//min(w, h))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        frame_list.append(frame)
    frames = np.stack(frame_list, axis=0)
    return frames

def _pascal_color_map(N=256, normalized=False):
    """
    Python implementation of the color map function for the PASCAL VOC data set.
    Official Matlab version can be found in the PASCAL VOC devkit
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
    """

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap

color_map = [
    [0, 0, 0], 
    [255, 50, 50], 
    [50, 255, 50], 
    [50, 50, 255], 
    [255, 50, 255], 
    [50, 255, 255], 
    [255, 255, 50], 
]

color_map_np = np.array(color_map)

def overlay_davis(image, mask, alpha=0.5):
    """ Overlay segmentation on top of RGB image. from davis official"""
    im_overlay = image.copy()

    colored_mask = color_map_np[mask]
    foreground = image*alpha + (1-alpha)*colored_mask
    binary_mask = (mask > 0)
    # Compose image
    im_overlay[binary_mask] = foreground[binary_mask]
    countours = binary_dilation(binary_mask) ^ binary_mask
    im_overlay[countours,:] = 0
    return im_overlay.astype(image.dtype)

def overlay_davis_fade(image, mask, alpha=0.5):
    im_overlay = image.copy()

    colored_mask = color_map_np[mask]
    foreground = image*alpha + (1-alpha)*colored_mask
    binary_mask = (mask > 0)
    # Compose image
    im_overlay[binary_mask] = foreground[binary_mask]
    countours = binary_dilation(binary_mask) ^ binary_mask
    im_overlay[countours,:] = 0
    im_overlay[~binary_mask] = im_overlay[~binary_mask] * 0.6
    return im_overlay.astype(image.dtype)