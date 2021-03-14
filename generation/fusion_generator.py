"""
A helper class for generating data used to train the fusion module.
This part is rather crude as it is used only once/twice.
"""
import torch
import numpy as np

from model.propagation.prop_net import PropagationNetwork
from model.aggregate import aggregate_sbg, aggregate_wbg

from util.tensor_util import pad_divide_by

class FusionGenerator:
    def __init__(self, prop_net:PropagationNetwork, images, mem_freq):
        self.prop_net = prop_net
        self.mem_freq = mem_freq

        # True dimensions
        t = images.shape[1]
        h, w = images.shape[-2:]

        # Pad each side to multiple of 16
        images, self.pad = pad_divide_by(images, 16, images.shape[-2:])
        # Padded dimensions
        nh, nw = images.shape[-2:]

        self.images = images
        self.device = self.images.device

        self.t, self.h, self.w = t, h, w
        self.nh, self.nw = nh, nw

    def reset(self, k):
        self.k = k
        self.prob = torch.zeros((self.k+1, self.t, 1, self.nh, self.nw), dtype=torch.float32, device=self.device)

    def get_im(self, idx):
        return self.images[:,idx]

    def get_query_buf(self, idx):
        query = self.prop_net.get_query_values(self.get_im(idx))
        return query

    def do_pass(self, key_k, key_v, idx, left_limit, right_limit, forward=True):
        keys = key_k
        values = key_v
        prev_k = prev_v = None
        last_ti = idx
        
        Es = self.prob

        if forward:
            this_range = range(idx+1, right_limit+1)
            step = +1
            end = right_limit
        else:
            this_range = range(idx-1, left_limit-1, -1)
            step = -1
            end = left_limit

        for ti in this_range:
            if prev_k is not None:
                this_k = torch.cat([keys, prev_k], 2)
                this_v = torch.cat([values, prev_v], 2)
            else:
                this_k = keys
                this_v = values
            query = self.get_query_buf(ti)
            out_mask = self.prop_net.segment_with_query(this_k, this_v, *query)
            out_mask = aggregate_wbg(out_mask, keep_bg=True)

            Es[:,ti] = out_mask

            if ti != end:
                prev_k, prev_v = self.prop_net.memorize(self.get_im(ti), out_mask[1:])
                if abs(ti-last_ti) >= self.mem_freq:
                    last_ti = ti
                    keys = torch.cat([keys, prev_k], 2)
                    values = torch.cat([values, prev_v], 2)
                    prev_k = prev_v = None

    def interact_mask(self, mask, idx, left_limit, right_limit):

        mask, _ = pad_divide_by(mask, 16, mask.shape[-2:])
        mask = aggregate_wbg(mask, keep_bg=True)

        self.prob[:, idx] = mask
        key_k, key_v = self.prop_net.memorize(self.get_im(idx), mask[1:])

        self.do_pass(key_k, key_v, idx, left_limit, right_limit, True)
        self.do_pass(key_k, key_v, idx, left_limit, right_limit, False)

        # Prepare output
        out_prob = self.prob[:,:,0,:,:]

        if self.pad[2]+self.pad[3] > 0:
            out_prob = out_prob[:,:,self.pad[2]:-self.pad[3],:]
        if self.pad[0]+self.pad[1] > 0:
            out_prob = out_prob[:,:,:,self.pad[0]:-self.pad[1]]

        return out_prob
