import torch
import torch.nn as nn
import torch.nn.functional as F
from util.tensor_util import compute_tensor_iu
from collections import defaultdict


def get_iou_hook(values):
    return 'iou/iou', (values['hide_iou/i']+1)/(values['hide_iou/u']+1)

def get_sec_iou_hook(values):
    return 'iou/sec_iou', (values['hide_iou/sec_i']+1)/(values['hide_iou/sec_u']+1)

iou_hooks = [
    get_iou_hook,
    get_sec_iou_hook,
]


# https://stackoverflow.com/questions/63735255/how-do-i-compute-bootstrapped-cross-entropy-loss-in-pytorch
class BootstrappedCE(nn.Module):
    def __init__(self, start_warm, end_warm, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p

    def forward(self, input, target, it):
        if it < self.start_warm:
            return F.cross_entropy(input, target), 1.0

        raw_loss = F.cross_entropy(input, target, reduction='none').view(-1)
        num_pixels = raw_loss.numel()

        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1-self.top_p)*((self.end_warm-it)/(self.end_warm-self.start_warm))
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        return loss.mean(), this_p


class LossComputer:
    def __init__(self, para):
        super().__init__()
        self.para = para
        self.bce = BootstrappedCE(start_warm=int(para['iterations']*0.2), end_warm=int(para['iterations']*0.5))

    def compute(self, data, it):
        losses = defaultdict(int)

        b = data['gt'].shape[0]
        selector = data.get('selector', None)
        selector = data['selector']

        for j in range(b):
            if selector[j][1] > 0.5:
                loss, p = self.bce(data['logits'][j:j+1], data['cls_gt'][j:j+1], it)
            else:
                loss, p = self.bce(data['logits'][j:j+1,:2], data['cls_gt'][j:j+1], it)

            losses['total_loss'] += loss / b
            losses['p'] += p / b

        new_total_i, new_total_u = compute_tensor_iu(data['mask'][:,1:2]>0.5, data['gt']>0.5)
        losses['hide_iou/i'] += new_total_i
        losses['hide_iou/u'] += new_total_u

        if selector is not None:
            new_total_i, new_total_u = compute_tensor_iu(data['mask'][:,2:3]>0.5, data['gt2']>0.5)
            losses['hide_iou/sec_i'] += new_total_i
            losses['hide_iou/sec_u'] += new_total_u

        return losses