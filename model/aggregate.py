import torch
import torch.nn.functional as F

def aggregate_sbg(prob, keep_bg=False, hard=False):
    device = prob.device
    k, _, h, w = prob.shape
    ex_prob = torch.zeros((k+1, 1, h, w), device=device)
    ex_prob[0] = 0.5
    ex_prob[1:] = prob
    ex_prob = torch.clamp(ex_prob, 1e-7, 1-1e-7)
    logits = torch.log((ex_prob /(1-ex_prob)))

    if hard:
        # Very low temperature o((⊙﹏⊙))o 🥶
        logits *= 1000

    if keep_bg:
        return F.softmax(logits, dim=0)
    else:
        return F.softmax(logits, dim=0)[1:]

def aggregate_wbg(prob, keep_bg=False, hard=False):
    k, _, h, w = prob.shape
    new_prob = torch.cat([
        torch.prod(1-prob, dim=0, keepdim=True),
        prob
    ], 0).clamp(1e-7, 1-1e-7)
    logits = torch.log((new_prob /(1-new_prob)))

    if hard:
        # Very low temperature o((⊙﹏⊙))o 🥶
        logits *= 1000

    if keep_bg:
        return F.softmax(logits, dim=0)
    else:
        return F.softmax(logits, dim=0)[1:]

def aggregate_wbg_channel(prob, keep_bg=False, hard=False):
    new_prob = torch.cat([
        torch.prod(1-prob, dim=1, keepdim=True),
        prob
    ], 1).clamp(1e-7, 1-1e-7)
    logits = torch.log((new_prob /(1-new_prob)))

    if hard:
        # Very low temperature o((⊙﹏⊙))o 🥶
        logits *= 1000

    if keep_bg:
        return logits, F.softmax(logits, dim=1)
    else:
        return logits, F.softmax(logits, dim=1)[:, 1:]