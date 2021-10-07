import os
from os import path
from argparse import ArgumentParser

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import cv2

from model.propagation.prop_net import PropagationNetwork
from model.fusion_net import FusionNet
from model.s2m.s2m_network import deeplabv3plus_resnet50 as S2M
from dataset.davis_test_dataset import DAVISTestDataset
from davis_processor import DAVISProcessor

from davisinteractive.session.session import DavisInteractiveSession

"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--prop_model', default='saves/propagation_model.pth')
parser.add_argument('--fusion_model', default='saves/fusion.pth')
parser.add_argument('--s2m_model', default='saves/s2m.pth')
parser.add_argument('--davis', default='../DAVIS/2017')
parser.add_argument('--output')
parser.add_argument('--save_mask', action='store_true')

args = parser.parse_args()

davis_path = args.davis
out_path = args.output
save_mask = args.save_mask

# Simple setup
os.makedirs(out_path, exist_ok=True)
palette = Image.open(path.expanduser(davis_path + '/trainval/Annotations/480p/blackswan/00000.png')).getpalette()

torch.autograd.set_grad_enabled(False)

# Setup Dataset
test_dataset = DAVISTestDataset(davis_path+'/trainval', imset='2017/val.txt')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

images = {}
num_objects = {}
# Loads all the images
for data in test_loader:
    rgb = data['rgb']
    k = len(data['info']['labels'][0])
    name = data['info']['name'][0]
    images[name] = rgb
    num_objects[name] = k
print('Finished loading %d sequences.' % len(images))

# Load our checkpoint
prop_saved = torch.load(args.prop_model)
prop_model = PropagationNetwork().cuda().eval()
prop_model.load_state_dict(prop_saved)

fusion_saved = torch.load(args.fusion_model)
fusion_model = FusionNet().cuda().eval()
fusion_model.load_state_dict(fusion_saved)

s2m_saved = torch.load(args.s2m_model)
s2m_model = S2M().cuda().eval()
s2m_model.load_state_dict(s2m_saved)

total_iter = 0
user_iter = 0
last_seq = None
pred_masks = None
with DavisInteractiveSession(davis_root=davis_path+'/trainval', report_save_dir='../output', max_nb_interactions=8, max_time=8*30) as sess:
    while sess.next():
        sequence, scribbles, new_seq = sess.get_scribbles(only_last=True)

        if new_seq:
            if 'processor' in locals():
                # Note that ALL pre-computed features are flushed in this step
                # We are not using pre-computed features for the same sequence with different user-id
                del processor # Should release some juicy mem
            processor = DAVISProcessor(prop_model, fusion_model, s2m_model, images[sequence], num_objects[sequence])
            print(sequence)

            # Save last time
            if save_mask:
                if pred_masks is not None:
                    seq_path = path.join(out_path, str(user_iter), last_seq)
                    os.makedirs(seq_path, exist_ok=True)
                    for i in range(len(pred_masks)):
                        img_E = Image.fromarray(pred_masks[i])
                        img_E.putpalette(palette)
                        img_E.save(os.path.join(seq_path, '{:05d}.png'.format(i)))

                if (last_seq is None) or (sequence != last_seq):
                    last_seq = sequence
                    user_iter = 0
                else:
                    user_iter += 1

        pred_masks, next_masks, this_idx = processor.interact(scribbles)
        sess.submit_masks(pred_masks, next_masks)

        total_iter += 1

    report = sess.get_report()
    summary = sess.get_global_summary(save_file=path.join(out_path, 'summary.json'))
