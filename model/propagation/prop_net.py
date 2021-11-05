"""
Modifed from the original STM code https://github.com/seoungwugoh/STM
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from model.propagation.modules import *


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = ResBlock(1024, 512)
        self.up_16_8 = UpsampleBlock(512, 512, 256) # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(256, 256, 256) # 1/8 -> 1/4

        self.pred = nn.Conv2d(256, 1, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, f16, f8, f4):
        x = self.compress(f16)
        x = self.up_16_8(f8, x)
        x = self.up_8_4(f4, x)

        x = self.pred(F.relu(x))
        
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return x


def make_gaussian(y_idx, x_idx, height, width, sigma=7):
    yv, xv = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])

    yv = yv.reshape(height*width).unsqueeze(0).float().cuda()
    xv = xv.reshape(height*width).unsqueeze(0).float().cuda()

    y_idx = y_idx.transpose(0, 1)
    x_idx = x_idx.transpose(0, 1)

    g = torch.exp(- ((yv-y_idx)**2 + (xv-x_idx)**2) / (2*sigma**2) )

    return g

def softmax_w_g_top(x, top=None, gauss=None):
    if top is not None:
        if gauss is not None:
            maxes = torch.max(x, dim=1, keepdim=True)[0]
            x_exp = torch.exp(x - maxes)*gauss
            x_exp, indices = torch.topk(x_exp, k=top, dim=1)
        else:
            values, indices = torch.topk(x, k=top, dim=1)
            x_exp = torch.exp(values - values[:,0])

        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        x_exp /= x_exp_sum
        # The types should be the same already
        # some people report an error here so an additional guard is added
        x.zero_().scatter_(1, indices, x_exp.type(x.dtype)) # B * THW * HW

        output = x
    else:
        maxes = torch.max(x, dim=1, keepdim=True)[0]
        if gauss is not None:
            x_exp = torch.exp(x-maxes)*gauss

        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        x_exp /= x_exp_sum
        output = x_exp

    return output

class EvalMemoryReader(nn.Module):
    def __init__(self, top_k, km):
        super().__init__()
        self.top_k = top_k
        self.km = km

    def forward(self, mk, mv, qk):
        B, CK, T, H, W = mk.shape
        _, CV, _, _, _ = mv.shape

        mi = mk.view(B, CK, T*H*W).transpose(1, 2)
        qi = qk.view(1, CK, H*W).expand(B, -1, -1) / math.sqrt(CK)  # B * CK * HW
 
        affinity = torch.bmm(mi, qi)  # B, THW, HW

        if self.km is not None:
            # Make a bunch of Gaussian distributions
            argmax_idx = affinity.max(2)[1]
            y_idx, x_idx = argmax_idx//W, argmax_idx%W
            g = make_gaussian(y_idx, x_idx, H, W, sigma=self.km)
            g = g.view(B, T*H*W, H*W)

            affinity = softmax_w_g_top(affinity, top=self.top_k, gauss=g)  # B, THW, HW
        else:
            if self.top_k is not None:
                affinity = softmax_w_g_top(affinity, top=self.top_k, gauss=None)  # B, THW, HW
            else:
                affinity = F.softmax(affinity, dim=1)

        mo = mv.view(B, CV, T*H*W) 
        mem = torch.bmm(mo, affinity) # Weighted-sum B, CV, HW
        mem = mem.view(B, CV, H, W)

        return mem

class AttentionMemory(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
 
    def forward(self, mk, qk): 
        """
        T=1 only. Only needs to obtain W
        """
        B, CK, _, H, W = mk.shape

        mk = mk.view(B, CK, H*W) 
        mk = torch.transpose(mk, 1, 2)  # B * HW * CK
 
        qk = qk.view(1, CK, H*W).expand(B, -1, -1) / math.sqrt(CK)  # B * CK * HW
 
        affinity = torch.bmm(mk, qk) # B * HW * HW
        affinity = F.softmax(affinity, dim=1)

        return affinity

class PropagationNetwork(nn.Module):
    def __init__(self, top_k=50):
        super().__init__()
        self.mask_rgb_encoder = MaskRGBEncoder() 
        self.rgb_encoder = RGBEncoder() 

        self.kv_m_f16 = KeyValue(1024, keydim=128, valdim=512)
        self.kv_q_f16 = KeyValue(1024, keydim=128, valdim=512)

        self.memory = EvalMemoryReader(top_k, km=None)
        self.attn_memory = AttentionMemory(top_k)
        self.decoder = Decoder()

    def memorize(self, frame, masks): 
        k, _, h, w = masks.shape

        # Extract memory key/value for a frame with multiple masks
        frame = frame.view(1, 3, h, w).repeat(k, 1, 1, 1)
        # Compute the "others" mask
        if k != 1:
            others = torch.cat([
                torch.sum(
                    masks[[j for j in range(k) if i!=j]]
                , dim=0, keepdim=True)
            for i in range(k)], 0)
        else:
            others = torch.zeros_like(masks)

        f16 = self.mask_rgb_encoder(frame, masks, others)
        k16, v16 = self.kv_m_f16(f16) # num_objects, 128 and 512, H/16, W/16

        return k16.unsqueeze(2), v16.unsqueeze(2)

    def get_query_values(self, frame):
        f16, f8, f4 = self.rgb_encoder(frame)
        k16, v16 = self.kv_q_f16(f16)  

        return f16, f8, f4, k16, v16

    def segment_with_query(self, keys, values, f16, f8, f4, k16, v16): 
        k = keys.shape[0]
        # Do it batch by batch to reduce memory usage
        batched = 1
        m4 = torch.cat([
            self.memory(keys[i:i+batched], values[i:i+batched], k16) for i in range(0, k, batched)
        ], 0)

        v16 = v16.expand(k, -1, -1, -1)
        m4 = torch.cat([m4, v16], 1)

        return torch.sigmoid(self.decoder(m4, f8, f4))

    def get_W(self, mk16, qk):
        W = self.attn_memory(mk16, qk)
        return W

    def get_attention(self, mk16, pos_mask, neg_mask, qk16):
        b, _, h, w = pos_mask.shape
        nh = h//16
        nw = w//16

        W = self.get_W(mk16, qk16)

        pos_map = (F.interpolate(pos_mask, size=(nh,nw), mode='area').view(b, 1, nh*nw) @ W)
        neg_map = (F.interpolate(neg_mask, size=(nh,nw), mode='area').view(b, 1, nh*nw) @ W)
        attn_map = torch.cat([pos_map, neg_map], 1)
        attn_map = attn_map.reshape(b, 2, nh, nw)
        attn_map = F.interpolate(attn_map, mode='bilinear', size=(h,w), align_corners=False)

        return attn_map