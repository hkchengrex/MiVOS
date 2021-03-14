"""
Modified from https://github.com/seoungwugoh/STM/blob/master/dataset.py
"""

import os
from os import path
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from dataset.range_transform import im_normalization
from dataset.onehot_util import all_to_onehot


class BLTestDataset(Dataset):
    def __init__(self, root, subset=None, start=None, end=None):
        self.root = root
        self.mask_dir = path.join(root, 'Annotations')
        self.image_dir = path.join(root, 'JPEGImages')

        self.videos = []
        self.num_frames = {}
        for _video in os.listdir(self.image_dir):
            if subset is not None and _video not in subset:
                continue

            self.videos.append(_video)
            self.num_frames[_video] = len(os.listdir(path.join(self.image_dir, _video)))

        self.im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])

        self.videos = sorted(self.videos)
        print('Total amount of videos: ', len(self.videos))
        if (start is not None) and (end is not None):
            print('Taking crop from %d to %d. ' % (start, end))
            self.videos = self.videos[start:end+1]
            print('New size: ', len(self.videos))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video = self.videos[index]
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]

        images = []
        masks = []
        for f in range(self.num_frames[video]):
            img_file = path.join(self.image_dir, video, '{:05d}.jpg'.format(f))
            images.append(self.im_transform(Image.open(img_file).convert('RGB')))
            
            mask_file = path.join(self.mask_dir, video, '{:05d}.png'.format(f))
            masks.append(np.array(Image.open(mask_file).convert('P'), dtype=np.uint8))
        
        images = torch.stack(images, 0)
        masks = np.stack(masks, 0)
        
        labels = np.unique(masks)
        labels = labels[labels!=0]
        masks = torch.from_numpy(all_to_onehot(masks, labels)).float()

        masks = masks.unsqueeze(2)

        info['labels'] = labels

        data = {
            'rgb': images,
            'gt': masks,
            'info': info,
        }

        return data

