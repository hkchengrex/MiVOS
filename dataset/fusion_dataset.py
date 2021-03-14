# Partially taken from STM's dataloader

import os
from os import path

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import random
from collections import defaultdict

from dataset.range_transform import im_normalization, im_mean
from dataset.reseed import reseed


class FusionDataset(Dataset):
    def __init__(self, im_root, gt_root, fd_root):
        """
        fd_root: Root to fusion_data/davis or fusion_data/bl

        Lots of varables here! See the return dict (at the end) for some comments
        """
        self.im_root = im_root
        self.gt_root = gt_root
        self.fd_root = fd_root

        self.videos = []
        self.frames = {}
        self.vid_to_instance = defaultdict(list)

        vid_list = sorted(os.listdir(self.im_root))
        for vid in vid_list:
            frames = sorted(os.listdir(os.path.join(self.im_root, vid)))
            self.frames[vid] = frames
            self.videos.append(vid)

        total_fuse_vid = 0
        fuse_list = sorted(os.listdir(self.fd_root))
        # run-level - different parameters
        for folder in fuse_list:
            folder_path = path.join(self.fd_root, folder)
            video_list = sorted(os.listdir(folder_path))
            # video level - different videos
            for vid in video_list:
                video_path = path.join(self.fd_root, folder, vid)
                self.vid_to_instance[vid].append(video_path)
                total_fuse_vid += 1

        # Filter out videos with no out
        self.videos = [v for v in self.videos if v in self.vid_to_instance]

        print('%d out of %d videos accepted.' % (len(self.videos), len(vid_list)))
        print('%d fusion videos accepted' % (total_fuse_vid))

        self.im_dual_transform = transforms.Compose([
            # transforms.RandomAffine(degrees=30, shear=10, fillcolor=im_mean, resample=Image.BILINEAR),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomResizedCrop((384, 384), scale=(0.34,1.0), ratio=(0.9,1.1), interpolation=Image.BILINEAR),
            transforms.RandomCrop(384),
            transforms.ColorJitter(0.1, 0.03, 0.03, 0.01),
        ])

        self.gt_dual_transform = transforms.Compose([
            # transforms.RandomAffine(degrees=30, shear=10, fillcolor=0, resample=Image.NEAREST),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomResizedCrop((384, 384), scale=(0.34,1.0), ratio=(0.9,1.1), interpolation=Image.NEAREST),
            transforms.RandomCrop(384),
        ])

        self.sg_dual_transform = transforms.Compose([
            # transforms.RandomAffine(degrees=30, shear=10, fillcolor=0, resample=Image.BILINEAR),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomResizedCrop((384, 384), scale=(0.34,1.0), ratio=(0.9,1.1), interpolation=Image.BILINEAR),
            transforms.RandomCrop(384),
        ])

        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            im_normalization,
        ])

    def __getitem__(self, idx):
        info = {}
        info['frames'] = [] # Appended with actual frames

        # Try a few times
        max_trial = 20
        for trials in range(max_trial):
            if trials < 5:
                video = self.videos[idx % len(self.videos)]
            else:
                video = np.random.choice(self.videos)
            vid_im_path = path.join(self.im_root, video)
            vid_gt_path = path.join(self.gt_root, video)
            info['name'] = video
            frames = self.frames[video]

            sequence_seed = np.random.randint(2147483647)
            video_path = self.vid_to_instance[video][np.random.choice(range(len(self.vid_to_instance[video])))]

            # Randomly pick the reference frames and object
            all_ref = os.listdir(video_path)
            first_ref = np.random.choice(all_ref)
            tar_obj = np.random.choice(os.listdir(path.join(video_path, first_ref)))
            tar_frame = np.random.choice(os.listdir(path.join(video_path, first_ref, tar_obj)))
            tar_obj_int = int(tar_obj)
            tar_frame_int = int(tar_frame[:-4])

            # Pick the second reference frame
            src2_ref_options = []
            for r in all_ref:
                # No self-referecne
                if r == first_ref:
                    continue
                # We need the second reference frame to be visible from the first
                if not path.exists(path.join(video_path, first_ref, tar_obj, r+'.png')):
                    continue
                # We need the target object to exist
                if path.exists(path.join(video_path, r, tar_obj, tar_frame)):
                    src2_ref_options.append(r)
            
            if len(src2_ref_options)>0:
                secon_ref = np.random.choice(src2_ref_options)
            else:
                continue

            # Pick another object that is valid in both reference frame
            sec_obj_options = [obj for obj in os.listdir(path.join(video_path, first_ref)) 
                    if path.exists(path.join(video_path, first_ref, obj, tar_frame)) and
                        path.exists(path.join(video_path, secon_ref, obj, tar_frame)) and
                        obj != tar_obj]
            if len(sec_obj_options) == 0:
                sec_obj = -1
            else:
                sec_obj = np.random.choice(sec_obj_options)
            sec_obj_int = int(sec_obj)

            # Compute distance from reference frame to target frame
            dist_1 = abs(int(first_ref)-tar_frame_int) / abs(int(first_ref)-int(secon_ref))
            dist_2 = abs(int(secon_ref)-tar_frame_int) / abs(int(first_ref)-int(secon_ref))

            png_name = '%05d'%tar_frame_int + '.png'
            jpg_name = '%05d'%tar_frame_int + '.jpg'
            src2_ref_png_name = '%05d'%int(secon_ref) + '.png'
            src2_ref_jpg_name = '%05d'%int(secon_ref) + '.jpg'

            src1_seg = Image.open(path.join(video_path, first_ref, tar_obj, png_name)).convert('L')
            src2_seg = Image.open(path.join(video_path, secon_ref, tar_obj, png_name)).convert('L')

            # Transform these first two
            reseed(sequence_seed)
            src1_seg = np.array(self.sg_dual_transform(src1_seg))[:,:,np.newaxis]
            reseed(sequence_seed)
            src2_seg = np.array(self.sg_dual_transform(src2_seg))[:,:,np.newaxis]

            diff = np.abs(src1_seg.astype(np.float32) - src2_seg.astype(np.float32)) > (255*0.1)
            diff = diff.astype(np.uint8)
            usable_i, usable_j = np.nonzero(diff[:,:,0])
            if trials<max_trial*0.75 and len(usable_i) < 100:
                continue

            # Continue loading and transforming if they are OK
            src2_ref_seg = Image.open(path.join(video_path, first_ref, tar_obj, src2_ref_png_name)).convert('L')
            src2_ref_gt = Image.open(path.join(vid_gt_path, src2_ref_png_name)).convert('P')
            src2_ref_im = Image.open(path.join(vid_im_path, src2_ref_jpg_name)).convert('RGB')
            gt = Image.open(path.join(vid_gt_path, png_name)).convert('P')
            im = Image.open(path.join(vid_im_path, jpg_name)).convert('RGB')

            # Loads stuff for the second object
            if sec_obj != -1:
                src1_seg2 = Image.open(path.join(video_path, first_ref, sec_obj, tar_frame)).convert('L')
                src2_seg2 = Image.open(path.join(video_path, secon_ref, sec_obj, tar_frame)).convert('L')
                reseed(sequence_seed)
                src1_seg2 = np.array(self.sg_dual_transform(src1_seg2))[:,:,np.newaxis]
                reseed(sequence_seed)
                src2_seg2 = np.array(self.sg_dual_transform(src2_seg2))[:,:,np.newaxis]
                src2_ref_seg2 = Image.open(path.join(video_path, first_ref, sec_obj, src2_ref_png_name)).convert('L')

            reseed(sequence_seed)
            src2_ref_seg = np.array(self.sg_dual_transform(src2_ref_seg))[:,:,np.newaxis]
            reseed(sequence_seed)
            gt_mask = (np.array(self.gt_dual_transform(gt)) == tar_obj_int).astype(np.uint8)[:,:,np.newaxis]
            reseed(sequence_seed)
            src2_ref_mask = (np.array(self.gt_dual_transform(src2_ref_gt)) == tar_obj_int).astype(np.uint8)[:,:,np.newaxis]
            reseed(sequence_seed)
            im = np.array(self.im_dual_transform(im))
            reseed(sequence_seed)
            src2_ref_im = np.array(self.im_dual_transform(src2_ref_im))

            # For the second object
            if sec_obj != -1:
                reseed(sequence_seed)
                src2_ref_seg2 = np.array(self.sg_dual_transform(src2_ref_seg2))[:,:,np.newaxis]
                reseed(sequence_seed)
                gt_mask2 = (np.array(self.gt_dual_transform(gt)) == sec_obj_int).astype(np.uint8)[:,:,np.newaxis]
                reseed(sequence_seed)
                src2_ref_mask2 = (np.array(self.gt_dual_transform(src2_ref_gt)) == sec_obj_int).astype(np.uint8)[:,:,np.newaxis]

            break # all ok

        im = self.final_im_transform(torch.from_numpy(im.astype(np.float32)/255).permute(2,0,1))
        src2_ref_im = self.final_im_transform(torch.from_numpy(src2_ref_im.astype(np.float32)/255).permute(2,0,1))
        gt_mask = torch.from_numpy(gt_mask.astype(np.float32)).permute(2,0,1)
        src2_ref_mask = torch.from_numpy(src2_ref_mask.astype(np.float32)).permute(2,0,1)

        src1_seg = torch.from_numpy(src1_seg.astype(np.float32)/255).permute(2,0,1)
        src2_seg = torch.from_numpy(src2_seg.astype(np.float32)/255).permute(2,0,1)
        src2_ref_seg = torch.from_numpy(src2_ref_seg.astype(np.float32)/255).permute(2,0,1)

        if sec_obj != -1:
            gt_mask2 = torch.from_numpy(gt_mask2.astype(np.float32)).permute(2,0,1)
            src2_ref_mask2 = torch.from_numpy(src2_ref_mask2.astype(np.float32)).permute(2,0,1)

            src1_seg2 = torch.from_numpy(src1_seg2.astype(np.float32)/255).permute(2,0,1)
            src2_seg2 = torch.from_numpy(src2_seg2.astype(np.float32)/255).permute(2,0,1)
            src2_ref_seg2 = torch.from_numpy(src2_ref_seg2.astype(np.float32)/255).permute(2,0,1)

            selector = torch.FloatTensor([1, 1])
        else:
            gt_mask2 = torch.zeros_like(gt_mask)
            src2_ref_mask2 = torch.zeros_like(src2_ref_mask)
            src1_seg2 = torch.zeros_like(src1_seg)
            src2_seg2 = torch.zeros_like(src2_seg)
            src2_ref_seg2 = torch.zeros_like(src2_ref_seg)

            selector = torch.FloatTensor([1, 0])

        dist = torch.FloatTensor([dist_1, dist_2])

        cls_gt = np.zeros((384, 384), dtype=np.int)
        cls_gt[gt_mask[0] > 0.5] = 1
        cls_gt[gt_mask2[0] > 0.5] = 2

        data = {
            # Target frame is defined to be the frame that requires fusion
            'rgb': im, # Target frame image
            'cls_gt': cls_gt, # Target frame ground truth in int format

            # First object
            'gt': gt_mask, # GT mask of object 1 at the target frame
            'seg1': src1_seg, # Propagated mask from reference 1 of object 1 at the target frame
            'seg2': src2_seg, # Propagated mask from reference 2 of object 1 at the target frame
            'src2_ref': src2_ref_seg, # Propagated mask from reference 1 of object 1 at reference 2
            'src2_ref_gt': src2_ref_mask, # GT mask of object 1 at reference 2

            # Second object
            'gt2': gt_mask2, # GT mask of object 2 at the target frame
            'seg12': src1_seg2, # ... of object 2 ...
            'seg22': src2_seg2, # ... of object 2 ...
            'src2_ref2': src2_ref_seg2, # ... of object 2 ...
            'src2_ref_gt2': src2_ref_mask2, # ... of object 2 ...

            'src2_ref_im': src2_ref_im, # Image at reference 2
            'dist': dist,
            'selector': selector,
            'info': info,
        }

        return data

    def __len__(self):
        return len(self.videos) * 100