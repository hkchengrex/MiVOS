"""
Modifed from the original STM code https://github.com/seoungwugoh/STM
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.propagation.modules import *

class AttentionMemory(nn.Module):
    def __init__(self, k=50):
        super().__init__()
        self.k = k
 
    def forward(self, mk, qk): 
        B, CK, H, W = mk.shape

        mk = mk.flatten(start_dim=2)
        qk = qk.flatten(start_dim=2)

        a = mk.pow(2).sum(1).unsqueeze(2)
        b = 2 * (mk.transpose(1, 2) @ qk)
        c = qk.pow(2).sum(1).unsqueeze(1)

        affinity = (-a+b-c) / math.sqrt(CK)   # B, THW, HW

        affinity = F.softmax(affinity, dim=1)

        return affinity

class AttentionReadNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.key_encoder = KeyEncoder() 
        self.key_proj = KeyProjection(1024, keydim=64)

        self.memory = AttentionMemory()

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, image, mask11, mask21, mask12, mask22, query_image):
        b, _, h, w = mask11.shape
        nh = h//16
        nw = w//16
        
        with torch.no_grad():
            pos_mask1 = (mask21-mask11).clamp(0, 1)
            neg_mask1 = (mask11-mask21).clamp(0, 1)
            pos_mask2 = (mask22-mask12).clamp(0, 1)
            neg_mask2 = (mask12-mask22).clamp(0, 1)

            qk16 = self.key_proj(self.key_encoder(query_image)[0])
            mk16 = self.key_proj(self.key_encoder(image)[0])

            W = self.memory(mk16, qk16)

            pos_map1 = (F.interpolate(pos_mask1, size=(nh,nw), mode='area').view(b, 1, nh*nw) @ W)
            neg_map1 = (F.interpolate(neg_mask1, size=(nh,nw), mode='area').view(b, 1, nh*nw) @ W)
            attn_map1 = torch.cat([pos_map1, neg_map1], 1)
            attn_map1 = attn_map1.reshape(b, 2, nh, nw)
            attn_map1 = F.interpolate(attn_map1, mode='bilinear', size=(h,w), align_corners=False)

            pos_map2 = (F.interpolate(pos_mask2, size=(nh,nw), mode='area').view(b, 1, nh*nw) @ W)
            neg_map2 = (F.interpolate(neg_mask2, size=(nh,nw), mode='area').view(b, 1, nh*nw) @ W)
            attn_map2 = torch.cat([pos_map2, neg_map2], 1)
            attn_map2 = attn_map2.reshape(b, 2, nh, nw)
            attn_map2 = F.interpolate(attn_map2, mode='bilinear', size=(h,w), align_corners=False)

        return attn_map1, attn_map2
