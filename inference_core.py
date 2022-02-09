"""
Heart of most evaluation scripts (DAVIS semi-sup/interactive, GUI)
Handles propagation and fusion
See eval_semi_davis.py / eval_interactive_davis.py for examples
"""

import torch
import numpy as np
import cv2

from model.propagation.prop_net import PropagationNetwork
from model.fusion_net import FusionNet
from model.aggregate import aggregate_sbg, aggregate_wbg

from util.tensor_util import pad_divide_by

class InferenceCore:
    """
    images - leave them in original dimension (unpadded), but do normalize them. 
            Should be CPU tensors of shape B*T*3*H*W
            
    mem_profile - How extravagant I can use the GPU memory. 
                Usually more memory -> faster speed but I have not drawn the exact relation
                0 - Use the most memory
                1 - Intermediate, larger buffer 
                2 - Intermediate, small buffer 
                3 - Use the minimal amount of GPU memory
                Note that *none* of the above options will affect the accuracy
                This is a space-time tradeoff, not a space-performance one

    mem_freq - Period at which new memory are put in the bank
                Higher number -> less memory usage
                Unlike the last option, this *is* a space-performance tradeoff
    """
    def __init__(self, prop_net:PropagationNetwork, fuse_net:FusionNet, images, num_objects, 
                    mem_profile=0, mem_freq=5, device='cuda:0'):
        self.prop_net = prop_net.to(device, non_blocking=True)
        if fuse_net is not None:
            self.fuse_net = fuse_net.to(device, non_blocking=True)
        self.mem_profile = mem_profile
        self.mem_freq = mem_freq
        self.device = device

        if mem_profile == 0:
            self.data_dev = device
            self.result_dev = device
            self.q_buf_size = 105
            self.i_buf_size = -1 # no need to buffer image
        elif mem_profile == 1:
            self.data_dev = 'cpu'
            self.result_dev = device
            self.q_buf_size = 105
            self.i_buf_size = 105
        elif mem_profile == 2:
            self.data_dev = 'cpu'
            self.result_dev = 'cpu'
            self.q_buf_size = 3
            self.i_buf_size = 3
        else:
            self.data_dev = 'cpu'
            self.result_dev = 'cpu'
            self.q_buf_size = 1
            self.i_buf_size = 1

        # True dimensions
        t = images.shape[1]
        h, w = images.shape[-2:]
        self.k = num_objects

        # Pad each side to multiples of 16
        self.images, self.pad = pad_divide_by(images, 16, images.shape[-2:])
        # Padded dimensions
        nh, nw = self.images.shape[-2:]
        self.images = self.images.to(self.data_dev, non_blocking=False)

        # These two store the same information in different formats
        self.masks = torch.zeros((t, 1, nh, nw), dtype=torch.uint8, device=self.result_dev)
        self.np_masks = np.zeros((t, h, w), dtype=np.uint8)

        # Object probabilities, background included
        self.prob = torch.zeros((self.k+1, t, 1, nh, nw), dtype=torch.float32, device=self.result_dev)
        self.prob[0] = 1e-7

        self.t, self.h, self.w = t, h, w
        self.nh, self.nw = nh, nw
        self.kh = self.nh//16
        self.kw = self.nw//16

        self.query_buf = {}
        self.image_buf = {}
        self.interacted = set()

        self.certain_mem_k = None
        self.certain_mem_v = None

    def get_image_buffered(self, idx):
        if self.data_dev == self.device:
            return self.images[:,idx]

        # buffer the .cuda() calls
        if idx not in self.image_buf:
            # Flush buffer
            if len(self.image_buf) > self.i_buf_size:
                self.image_buf = {}
        self.image_buf[idx] = self.images[:,idx].to(self.device)
        result = self.image_buf[idx]

        return result

    def get_query_kv_buffered(self, idx):
        # Queries' key/value never change, so we can buffer them here
        if idx not in self.query_buf:
            # Flush buffer
            if len(self.query_buf) > self.q_buf_size:
                self.query_buf = {}

            self.query_buf[idx] = self.prop_net.get_query_values(self.get_image_buffered(idx))
        result = self.query_buf[idx]

        return result

    def do_pass(self, key_k, key_v, idx, forward=True, step_cb=None):
        """
        Do a complete pass that includes propagation and fusion
        key_k/key_v -  memory feature of the starting frame
        idx - Frame index of the starting frame
        forward - forward/backward propagation
        step_cb - Callback function used for GUI (progress bar) only
        """

        # Pointer in the memory bank
        num_certain_keys = self.certain_mem_k.shape[2]
        m_front = num_certain_keys

        # Determine the required size of the memory bank
        if forward:
            closest_ti = min([ti for ti in self.interacted if ti > idx] + [self.t])
            total_m = (closest_ti - idx - 1)//self.mem_freq + 1 + num_certain_keys
        else:
            closest_ti = max([ti for ti in self.interacted if ti < idx] + [-1])
            total_m = (idx - closest_ti - 1)//self.mem_freq + 1 + num_certain_keys
        K, CK, _, H, W = key_k.shape
        _, CV, _, _, _ = key_v.shape

        # Pre-allocate keys/values memory
        keys = torch.empty((K, CK, total_m, H, W), dtype=torch.float32, device=self.device)
        values = torch.empty((K, CV, total_m, H, W), dtype=torch.float32, device=self.device)

        # Initial key/value passed in
        keys[:,:,0:num_certain_keys] = self.certain_mem_k
        values[:,:,0:num_certain_keys] = self.certain_mem_v
        prev_in_mem = True
        last_ti = idx

        # Note that we never reach closest_ti, just the frame before it
        if forward:
            this_range = range(idx+1, closest_ti)
            step = +1
            end = closest_ti - 1
        else:
            this_range = range(idx-1, closest_ti, -1)
            step = -1
            end = closest_ti + 1

        for ti in this_range:
            if prev_in_mem:
                this_k = keys[:,:,:m_front]
                this_v = values[:,:,:m_front]
            else:
                this_k = keys[:,:,:m_front+1]
                this_v = values[:,:,:m_front+1]
            query = self.get_query_kv_buffered(ti)
            out_mask = self.prop_net.segment_with_query(this_k, this_v, *query)

            out_mask = aggregate_wbg(out_mask, keep_bg=True)

            if ti != end:
                keys[:,:,m_front:m_front+1], values[:,:,m_front:m_front+1] = self.prop_net.memorize(
                        self.get_image_buffered(ti), out_mask[1:])
                if abs(ti-last_ti) >= self.mem_freq:
                    # Memorize the frame
                    m_front += 1
                    last_ti = ti
                    prev_in_mem = True
                else:
                    prev_in_mem = False

            # In-place fusion, maximizes the use of queried buffer
            # esp. for long sequence where the buffer will be flushed
            if (closest_ti != self.t) and (closest_ti != -1):
                self.prob[:,ti] = self.fuse_one_frame(closest_ti, idx, ti, self.prob[:,ti], out_mask, 
                                        key_k, query[3]).to(self.result_dev)
            else:
                self.prob[:,ti] = out_mask.to(self.result_dev)

            # Callback function for the GUI
            if step_cb is not None:
                step_cb()

        return closest_ti

    def fuse_one_frame(self, tc, tr, ti, prev_mask, curr_mask, mk16, qk16):
        assert(tc<ti<tr or tr<ti<tc)

        prob = torch.zeros((self.k, 1, self.nh, self.nw), dtype=torch.float32, device=self.device)

        # Compute linear coefficients
        nc = abs(tc-ti) / abs(tc-tr)
        nr = abs(tr-ti) / abs(tc-tr)
        dist = torch.FloatTensor([nc, nr]).to(self.device).unsqueeze(0)
        for k in range(1, self.k+1):
            attn_map = self.prop_net.get_attention(mk16[k-1:k], self.pos_mask_diff[k:k+1], self.neg_mask_diff[k:k+1], qk16)

            w = torch.sigmoid(self.fuse_net(self.get_image_buffered(ti), 
                    prev_mask[k:k+1].to(self.device), curr_mask[k:k+1].to(self.device), attn_map, dist))
            prob[k-1] = w 
        return aggregate_wbg(prob, keep_bg=True)

    def interact(self, mask, idx, total_cb=None, step_cb=None):
        """
        Interact -> Propagate -> Fuse

        mask - One-hot mask of the interacted frame, background included
        idx - Frame index of the interacted frame
        total_cb, step_cb - Callback functions for the GUI

        Return: all mask results in np format for DAVIS evaluation
        """
        self.interacted.add(idx)

        mask = mask.to(self.device)
        mask, _ = pad_divide_by(mask, 16, mask.shape[-2:])
        self.mask_diff = mask - self.prob[:, idx].to(self.device)
        self.pos_mask_diff = self.mask_diff.clamp(0, 1)
        self.neg_mask_diff = (-self.mask_diff).clamp(0, 1)

        self.prob[:, idx] = mask
        key_k, key_v = self.prop_net.memorize(self.get_image_buffered(idx), mask[1:])

        if self.certain_mem_k is None:
            self.certain_mem_k = key_k
            self.certain_mem_v = key_v
        else:
            self.certain_mem_k = torch.cat([self.certain_mem_k, key_k], 2)
            self.certain_mem_v = torch.cat([self.certain_mem_v, key_v], 2)

        if total_cb is not None:
            # Finds the total num. frames to process
            front_limit = min([ti for ti in self.interacted if ti > idx] + [self.t])
            back_limit = max([ti for ti in self.interacted if ti < idx] + [-1])
            total_num = front_limit - back_limit - 2 # -1 for shift, -1 for center frame
            if total_num > 0:
                total_cb(total_num)

        self.do_pass(key_k, key_v, idx, True, step_cb=step_cb)
        self.do_pass(key_k, key_v, idx, False, step_cb=step_cb)
        
        # This is a more memory-efficient argmax
        for ti in range(self.t):
            self.masks[ti] = torch.argmax(self.prob[:,ti], dim=0)
        out_masks = self.masks

        # Trim paddings
        if self.pad[2]+self.pad[3] > 0:
            out_masks = out_masks[:,:,self.pad[2]:-self.pad[3],:]
        if self.pad[0]+self.pad[1] > 0:
            out_masks = out_masks[:,:,:,self.pad[0]:-self.pad[1]]

        self.np_masks = (out_masks.detach().cpu().numpy()[:,0]).astype(np.uint8)

        return self.np_masks

    def update_mask_only(self, prob_mask, idx):
        """
        Interaction only, no propagation/fusion
        prob_mask - mask of the interacted frame, background included
        idx - Frame index of the interacted frame

        Return: all mask results in np format for DAVIS evaluation
        """
        mask = torch.argmax(prob_mask, 0)
        self.masks[idx] = mask

        # Mask - 1 * H * W
        if self.pad[2]+self.pad[3] > 0:
            mask = mask[:,self.pad[2]:-self.pad[3],:]
        if self.pad[0]+self.pad[1] > 0:
            mask = mask[:,:,self.pad[0]:-self.pad[1]]

        mask = (mask.detach().cpu().numpy()[0]).astype(np.uint8)
        self.np_masks[idx] = mask

        return self.np_masks