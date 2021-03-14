import cv2
import numpy as np

import torch
import torchvision.transforms as transforms
from dataset.range_transform import inv_im_trans
from collections import defaultdict

def tensor_to_numpy(image):
    image_np = (image.numpy() * 255).astype('uint8')
    return image_np

def tensor_to_np_float(image):
    image_np = image.numpy().astype('float32')
    return image_np

def detach_to_cpu(x):
    return x.detach().cpu()

def transpose_np(x):
    return np.transpose(x, [1,2,0])

def tensor_to_gray_im(x):
    x = detach_to_cpu(x)
    x = tensor_to_numpy(x)
    x = transpose_np(x)
    return x

def tensor_to_im(x):
    x = detach_to_cpu(x)
    x = inv_im_trans(x).clamp(0, 1)
    x = tensor_to_numpy(x)
    x = transpose_np(x)
    return x

# Predefined key <-> caption dict
key_captions = {
    'im': 'Image', 
    'gt': 'GT', 
}

"""
Return an image array with captions
keys in dictionary will be used as caption if not provided
values should contain lists of cv2 images
"""
def get_image_array(images, grid_shape, captions={}):
    h, w = grid_shape
    cate_counts = len(images)
    rows_counts = len(next(iter(images.values())))

    font = cv2.FONT_HERSHEY_SIMPLEX

    output_image = np.zeros([w*cate_counts, h*(rows_counts+1), 3], dtype=np.uint8)
    col_cnt = 0
    for k, v in images.items():

        # Default as key value itself
        caption = captions.get(k, k)

        # Handles new line character
        dy = 40
        for i, line in enumerate(caption.split('\n')):
            if h > 200:
                cv2.putText(output_image, line, (10, col_cnt*w+100+i*dy),
                        font, 0.8, (255,255,255), 2, cv2.LINE_AA)
            else:
                cv2.putText(output_image, line, (10, col_cnt*w+10+i*dy),
                        font, 0.4, (255,255,255), 1, cv2.LINE_AA)

        # Put images
        for row_cnt, img in enumerate(v):
            im_shape = img.shape
            if len(im_shape) == 2:
                img = img[..., np.newaxis]

            img = (img * 255).astype('uint8')

            output_image[(col_cnt+0)*w:(col_cnt+1)*w,
                         (row_cnt+1)*h:(row_cnt+2)*h, :] = img
            
        col_cnt += 1

    return output_image

def base_transform(im, size):
        im = tensor_to_np_float(im)
        if len(im.shape) == 3:
            im = im.transpose((1, 2, 0))
        else:
            im = im[:, :, None]

        # Resize
        if size is not None and im.shape[1] != size:
            im = cv2.resize(im, size, interpolation=cv2.INTER_NEAREST)

        return im.clip(0, 1)

def im_transform(im, size):
    return base_transform(inv_im_trans(detach_to_cpu(im)), size=size)

def mask_transform(mask, size):
    return base_transform(detach_to_cpu(mask), size=size)

def out_transform(mask, size):
    return base_transform(detach_to_cpu(torch.sigmoid(mask)), size=size)

def get_click_points(image, pos_map, neg_map):
    image[pos_map<0.02, :] = [0, 1, 0]
    image[neg_map<0.02, :] = [1, 0, 0]

    return image

def get_clicked_torch(image, pos_map, neg_map):
    rgb = im_transform(image, None)
    pos_map = mask_transform(pos_map, None)[:, :, 0]
    neg_map = mask_transform(neg_map, None)[:, :, 0]

    rgb[pos_map<0.02, :] = [0, 1, 0]
    rgb[neg_map<0.02, :] = [1, 0, 0]

    return (rgb*255).astype(np.uint8)

def pool_fusion(images, size):
    req_images = defaultdict(list)

    b = images['gt'].shape[0]

    # Save storage
    b = max(4, b)

    GT_name = 'GT'

    for b_idx in range(b):
        req_images['RGB'].append(im_transform(images['rgb'][b_idx], size))
        req_images['S11'].append(mask_transform(images['seg1'][b_idx], size))
        req_images['S21'].append(mask_transform(images['seg2'][b_idx], size))
        req_images['S12'].append(mask_transform(images['seg12'][b_idx], size))
        req_images['S22'].append(mask_transform(images['seg22'][b_idx], size))
        req_images['Pos Attn1'].append(mask_transform(images['attn1'][b_idx,0:1], size))
        req_images['Neg Attn1'].append(mask_transform(images['attn1'][b_idx,1:2], size))
        req_images['Pos Attn2'].append(mask_transform(images['attn2'][b_idx,0:1], size))
        req_images['Neg Attn2'].append(mask_transform(images['attn2'][b_idx,1:2], size))

        req_images['MSK1'].append(mask_transform(images['mask'][b_idx,1:2], size))
        req_images['MSK2'].append(mask_transform(images['mask'][b_idx,2:3], size))

        req_images[GT_name+'1'].append(mask_transform(images['gt'][b_idx], size))
        req_images[GT_name+'2'].append(mask_transform(images['gt2'][b_idx], size))

    return get_image_array(req_images, size, key_captions)