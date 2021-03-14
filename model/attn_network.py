"""
Modifed from the original STM code https://github.com/seoungwugoh/STM
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.propagation.modules import MaskRGBEncoder, RGBEncoder, KeyValue

class AttentionMemory(nn.Module):
    def __init__(self, k=50):
        super().__init__()
        self.k = k
 
    def forward(self, mk, qk): 
        B, CK, H, W = mk.shape

        mk = mk.view(B, CK, H*W) 
        mk = torch.transpose(mk, 1, 2)  # B * HW * CK
 
        qk = qk.view(B, CK, H*W).expand(B, -1, -1) / math.sqrt(CK)  # B * CK * HW
 
        affinity = torch.bmm(mk, qk) # B * HW * HW
        affinity = F.softmax(affinity, dim=1)

        return affinity

class AttentionReadNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_rgb_encoder = MaskRGBEncoder() 
        self.rgb_encoder = RGBEncoder() 

        self.kv_m_f16 = KeyValue(1024, keydim=128, valdim=512)
        self.kv_q_f16 = KeyValue(1024, keydim=128, valdim=512)
        self.memory = AttentionMemory()

        for p in self.parameters():
            p.requires_grad = False

    def get_segment(self, f16, qk):
        k16, _ = self.kv_m_f16(f16)
        p = self.memory(k16, qk)
        return p

    def forward(self, image, mask11, mask21, mask12, mask22, query_image):
        b, _, h, w = mask11.shape
        nh = h//16
        nw = w//16
        
        with torch.no_grad():
            pos_mask1 = (mask21-mask11).clamp(0, 1)
            neg_mask1 = (mask11-mask21).clamp(0, 1)
            pos_mask2 = (mask22-mask12).clamp(0, 1)
            neg_mask2 = (mask12-mask22).clamp(0, 1)

            f16_1 = self.mask_rgb_encoder(image, mask21, mask22)
            f16_2 = self.mask_rgb_encoder(image, mask22, mask21)

            qf16, _, _ = self.rgb_encoder(query_image)
            qk16, _ = self.kv_q_f16(qf16)

            W1 = self.get_segment(f16_1, qk16)
            W2 = self.get_segment(f16_2, qk16)

            pos_map1 = (F.interpolate(pos_mask1, size=(nh,nw), mode='area').view(b, 1, nh*nw) @ W1)
            neg_map1 = (F.interpolate(neg_mask1, size=(nh,nw), mode='area').view(b, 1, nh*nw) @ W1)
            attn_map1 = torch.cat([pos_map1, neg_map1], 1)
            attn_map1 = attn_map1.reshape(b, 2, nh, nw)
            attn_map1 = F.interpolate(attn_map1, mode='bilinear', size=(h,w), align_corners=False)

            pos_map2 = (F.interpolate(pos_mask2, size=(nh,nw), mode='area').view(b, 1, nh*nw) @ W2)
            neg_map2 = (F.interpolate(neg_mask2, size=(nh,nw), mode='area').view(b, 1, nh*nw) @ W2)
            attn_map2 = torch.cat([pos_map2, neg_map2], 1)
            attn_map2 = attn_map2.reshape(b, 2, nh, nw)
            attn_map2 = F.interpolate(attn_map2, mode='bilinear', size=(h,w), align_corners=False)

        return attn_map1, attn_map2
