"""
Generate fusion data for the DAVIS dataset.
"""

import os
from os import path
from argparse import ArgumentParser

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import cv2

from model.propagation.prop_net import PropagationNetwork
from dataset.davis_test_dataset import DAVISTestDataset
from dataset.bl_test_dataset import BLTestDataset
from generation.fusion_generator import FusionGenerator

from progressbar import progressbar


"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--model', default='saves/propagation_model.pth')
parser.add_argument('--davis_root', default='../DAVIS/2017')
parser.add_argument('--bl_root', default='../BL30K')
parser.add_argument('--dataset', help='DAVIS/BL')
parser.add_argument('--output')
parser.add_argument('--separation', default=None, type=int)
parser.add_argument('--range', default=None, type=int)
parser.add_argument('--mem_freq', default=None, type=int)
parser.add_argument('--start', default=None, type=int)
parser.add_argument('--end', default=None, type=int)
args = parser.parse_args()

davis_path = args.davis_root
bl_path = args.bl_root
out_path = args.output
dataset_option = args.dataset

# Simple setup
os.makedirs(out_path, exist_ok=True)
palette = Image.open(path.expanduser(davis_path+'/trainval/Annotations/480p/blackswan/00000.png')).getpalette()

torch.autograd.set_grad_enabled(False)

# Setup Dataset
if dataset_option == 'DAVIS':
    test_dataset = DAVISTestDataset(davis_path+'/trainval', imset='2017/train.txt')
elif dataset_option == 'BL':
    test_dataset = BLTestDataset(bl_path, start=args.start, end=args.end)
else:
    print('Use --dataset DAVIS or --dataset BL')
    raise NotImplementedError

# test_dataset = BLTestDataset(args.bl, start=args.start, end=args.end, subset=load_sub_bl())
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)

# Load our checkpoint
prop_saved = torch.load(args.model)
prop_model = PropagationNetwork().cuda().eval()
prop_model.load_state_dict(prop_saved)

# Start evaluation
for data in progressbar(test_loader, max_value=len(test_loader), redirect_stdout=True):

    rgb = data['rgb'].cuda()
    msk = data['gt'][0].cuda()
    info = data['info']

    total_t = rgb.shape[1]
    processor = FusionGenerator(prop_model, rgb, args.mem_freq)

    for frame in range(0, total_t, args.separation):

        usable_keys = []
        for k in range(msk.shape[0]):
            if (msk[k,frame] > 0.5).sum() > 10*10:
                usable_keys.append(k)
        if len(usable_keys) == 0:
            continue
        if len(usable_keys) > 5:
            # Memory limit
            usable_keys = usable_keys[:5]

        k = len(usable_keys)
        processor.reset(k)
        this_msk = msk[usable_keys]

        # Make this directory
        this_out_path = path.join(out_path, info['name'][0], '%05d'%frame)
        os.makedirs(this_out_path, exist_ok=True)

        # Propagate
        if dataset_option == 'DAVIS':
            left_limit = 0
            right_limit = total_t-1
        else:
            left_limit = max(0, frame-args.range)
            right_limit = min(total_t-1, frame+args.range)
        
        pred_range = range(left_limit, right_limit+1)
        out_probs = processor.interact_mask(this_msk[:,frame], frame, left_limit, right_limit)

        for kidx, obj_id in enumerate(usable_keys):
            obj_out_path = path.join(this_out_path, '%05d'%(obj_id+1))
            os.makedirs(obj_out_path, exist_ok=True)
            prob_Es = (out_probs[kidx+1]*255).cpu().numpy().astype(np.uint8)

            for f in pred_range:
                img_E = Image.fromarray(prob_Es[f])
                img_E.save(os.path.join(obj_out_path, '{:05d}.png'.format(f)))

        del out_probs

    print(info['name'][0])
    torch.cuda.empty_cache()