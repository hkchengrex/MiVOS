import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import thinplate as tps


def warp_image_cv(img, c_src, c_dst, dshape=None):
    dshape = dshape or img.shape
    theta = tps.tps_theta_from_points(c_src, c_dst, reduced=True)
    grid = tps.tps_grid(theta, c_dst, dshape)
    mapx, mapy = tps.tps_grid_to_remap(grid, img.shape)
    return cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)

img = cv2.imread('image.png')

c_src = np.array([
    [0.0, 0.0],
    [1., 0],
    [1, 1],
    [0, 1],
    [0.3, 0.3],
    [0.7, 0.7],
])

c_dst = np.array([
    [-0.2, -0.2],
    [1., 0],    
    [1, 1],
    [0, 1],
    [0.4, 0.4],
    [0.6, 0.6],
])

for i in range(5):
    c_this_dst = c_src*(5-i)/5 + c_dst*i/5
    warped = warp_image_cv(img, c_src, c_this_dst)
    cv2.imwrite('%d.png'%i, warped)