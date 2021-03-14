# Partially taken from STM's dataloader

import os
from os import path

import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import random

from dataset.range_transform import im_normalization

class YouTubeVOSTestDataset(Dataset):
    def __init__(self, data_root, split):
        self.image_dir = path.join(data_root, 'vos', 'all_frames', split, 'JPEGImages')
        self.mask_dir = path.join(data_root, 'vos', split, 'Annotations')

        self.videos = []
        self.shape = {}
        self.frames = {}

        vid_list = sorted(os.listdir(self.image_dir))
        # Pre-reading
        for vid in vid_list:
            frames = sorted(os.listdir(os.path.join(self.image_dir, vid)))
            self.frames[vid] = frames

            self.videos.append(vid)
            first_mask = os.listdir(path.join(self.mask_dir, vid))[0]
            _mask = np.array(Image.open(path.join(self.mask_dir, vid, first_mask)).convert("P"))
            self.shape[vid] = np.shape(_mask)

        self.im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])

    # From STM's code
    def To_onehot(self, mask, labels):
        M = np.zeros((len(labels), mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for k, l in enumerate(labels):
            M[k] = (mask == l).astype(np.uint8)
        return M
    
    def All_to_onehot(self, masks, labels):
        Ms = np.zeros((len(labels), masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
        for n in range(masks.shape[0]):
            Ms[:,n] = self.To_onehot(masks[n], labels)
        return Ms

    def __getitem__(self, idx):
        video = self.videos[idx]
        info = {}
        info['name'] = video
        info['num_objects'] = 0
        info['frames'] = self.frames[video] 
        info['size'] = self.shape[video] # Real sizes
        info['gt_obj'] = {} # Frames with labelled objects

        vid_im_path = path.join(self.image_dir, video)
        vid_gt_path = path.join(self.mask_dir, video)

        frames = self.frames[video]

        images = []
        masks = []
        for i, f in enumerate(frames):
            img = Image.open(path.join(vid_im_path, f)).convert('RGB')
            images.append(self.im_transform(img))
            
            mask_file = path.join(vid_gt_path, f.replace('.jpg','.png'))
            if path.exists(mask_file):
                masks.append(np.array(Image.open(mask_file).convert('P'), dtype=np.uint8))
                this_labels = np.unique(masks[-1])
                this_labels = this_labels[this_labels!=0]
                info['gt_obj'][i] = this_labels
            else:
                # Mask not exists -> nothing in it
                masks.append(np.zeros(self.shape[video]))
        
        images = torch.stack(images, 0)
        masks = np.stack(masks, 0)
        
        # Construct the forward and backward mapping table for labels
        labels = np.unique(masks).astype(np.uint8)
        labels = labels[labels!=0]
        info['label_convert'] = {}
        info['label_backward'] = {}
        idx = 1
        for l in labels:
            info['label_convert'][l] = idx
            info['label_backward'][idx] = l
            idx += 1
        masks = torch.from_numpy(self.All_to_onehot(masks, labels)).float()

        # images = images.unsqueeze(0)
        masks = masks.unsqueeze(2)

        # Resize to 480p
        h, w = masks.shape[-2:]
        if h > w:
            new_size = (h*480//w, 480)
        else:
            new_size = (480, w*480//h)
        images = F.interpolate(images, size=new_size, mode='bicubic', align_corners=False)
        masks = F.interpolate(masks, size=(1, *new_size), mode='nearest')

        info['labels'] = labels

        data = {
            'rgb': images,
            'gt': masks,
            'info': info,
        }

        return data

    def __len__(self):
        return len(self.videos)