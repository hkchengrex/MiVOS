import os
from os import path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

from model.fusion_net import FusionNet
from model.attn_network import AttentionReadNetwork
from model.aggregate import aggregate_wbg_channel
from model.losses import LossComputer, iou_hooks
from util.log_integrator import Integrator
from util.image_saver import pool_fusion


class FusionModel:
    def __init__(self, para, logger=None, save_path=None, local_rank=0, world_size=1, distributed=True):
        self.para = para
        self.local_rank = local_rank

        if distributed:
            self.net = nn.parallel.DistributedDataParallel(FusionNet().cuda(), 
                device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
        else:
            self.net = nn.DataParallel(
                FusionNet().cuda(), 
                device_ids=[local_rank], output_device=local_rank)

        self.prop_net = AttentionReadNetwork().eval().cuda()

        # Setup logger when local_rank=0
        self.logger = logger
        self.save_path = save_path
        if logger is not None:
            self.last_time = time.time()
        self.train_integrator = Integrator(self.logger, distributed=distributed, local_rank=local_rank, world_size=world_size)
        self.train_integrator.add_hook(iou_hooks)
        self.val_integrator = Integrator(self.logger, distributed=distributed, local_rank=local_rank, world_size=world_size)
        self.loss_computer = LossComputer(para)

        self.train()
        self.optimizer = optim.Adam(filter(
            lambda p: p.requires_grad, self.net.parameters()), lr=para['lr'], weight_decay=1e-7)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, para['steps'], para['gamma'])

        # Logging info
        self.report_interval = 100
        self.save_im_interval = 500
        self.save_model_interval = 5000
        if para['debug']:
            self.report_interval = self.save_im_interval = 1

    def do_pass(self, data, it=0):
        # No need to store the gradient outside training
        torch.set_grad_enabled(self._is_train)

        for k, v in data.items():
            if type(v) != list and type(v) != dict and type(v) != int:
                data[k] = v.cuda(non_blocking=True)

        # See fusion_dataset.py for variable definitions
        im = data['rgb']

        seg1 = data['seg1']
        seg2 = data['seg2']
        src2_ref = data['src2_ref']
        src2_ref_gt = data['src2_ref_gt']

        seg12 = data['seg12']
        seg22 = data['seg22']
        src2_ref2 = data['src2_ref2']
        src2_ref_gt2 = data['src2_ref_gt2']

        src2_ref_im = data['src2_ref_im']
        selector = data['selector']
        dist = data['dist']

        out = {}
        # Get kernelized memory
        with torch.no_grad():
            attn1, attn2 = self.prop_net(src2_ref_im, src2_ref, src2_ref_gt, src2_ref2, src2_ref_gt2, im)

        prob1 = torch.sigmoid(self.net(im, seg1, seg2, attn1, dist))
        prob2 = torch.sigmoid(self.net(im, seg12, seg22, attn2, dist))
        prob = torch.cat([prob1, prob2], 1) * selector.unsqueeze(2).unsqueeze(2)
        logits, prob = aggregate_wbg_channel(prob, True)

        out['logits'] = logits
        out['mask'] = prob
        out['attn1'] = attn1
        out['attn2'] = attn2

        if self._do_log or self._is_train:
            losses = self.loss_computer.compute({**data, **out}, it)

            # Logging
            if self._do_log:
                self.integrator.add_dict(losses)
                if self._is_train:
                    if it % self.save_im_interval == 0 and it != 0:
                        if self.logger is not None:
                            images = {**data, **out}
                            size = (320, 320)
                            self.logger.log_cv2('train/pairs', pool_fusion(images, size=size), it)
                else:
                    # Validation save
                    if data['val_iter'] % 10 == 0:
                        if self.logger is not None:
                            images = {**data, **out}
                            size = (320, 320)
                            self.logger.log_cv2('val/pairs', pool_fusion(images, size=size), it)

        if self._is_train:
            if (it) % self.report_interval == 0 and it != 0:
                if self.logger is not None:
                    self.logger.log_scalar('train/lr', self.scheduler.get_last_lr()[0], it)
                    self.logger.log_metrics('train', 'time', (time.time()-self.last_time)/self.report_interval, it)
                self.last_time = time.time()
                self.train_integrator.finalize('train', it)
                self.train_integrator.reset_except_hooks()

            if it % self.save_model_interval == 0 and it != 0:
                if self.logger is not None:
                    self.save(it)

            # Backward pass
            self.optimizer.zero_grad(set_to_none=True) 
            losses['total_loss'].backward() 
            self.optimizer.step()
            self.scheduler.step()

    def save(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        model_path = self.save_path + ('_%s.pth' % it)
        torch.save(self.net.module.state_dict(), model_path)
        print('Model saved to %s.' % model_path)

        self.save_checkpoint(it)

    def save_checkpoint(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        checkpoint_path = self.save_path + '_checkpoint.pth'
        checkpoint = { 
            'it': it,
            'network': self.net.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}
        torch.save(checkpoint, checkpoint_path)

        print('Checkpoint saved to %s.' % checkpoint_path)

    def load_model(self, path):
        map_location = 'cuda:%d' % self.local_rank
        checkpoint = torch.load(path, map_location={'cuda:0': map_location})

        it = checkpoint['it']
        network = checkpoint['network']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']

        map_location = 'cuda:%d' % self.local_rank
        self.net.module.load_state_dict(network)
        self.optimizer.load_state_dict(optimizer)
        self.scheduler.load_state_dict(scheduler)

        print('Model loaded.')

        return it

    def load_network(self, path):
        map_location = 'cuda:%d' % self.local_rank
        self.net.module.load_state_dict(torch.load(path, map_location={'cuda:0': map_location}))
        # self.net.load_state_dict(torch.load(path))
        print('Network weight loaded:', path)

    def load_prop(self, path):
        map_location = 'cuda:%d' % self.local_rank
        self.prop_net.load_state_dict(torch.load(path, map_location={'cuda:0': map_location}), strict=False)
        print('Propagation network weight loaded:', path)

    def finalize_val(self, it):
        self.val_integrator.finalize('val', it)
        self.val_integrator.reset_except_hooks()

    def train(self):
        self._is_train = True
        self._do_log = True
        self.integrator = self.train_integrator
        # Also skip BN
        self.net.eval()
        self.prop_net.eval()
        return self

    def val(self):
        self._is_train = False
        self.integrator = self.val_integrator
        self._do_log = True
        self.net.eval()
        self.prop_net.eval()
        return self

    def test(self):
        self._is_train = False
        self._do_log = False
        self.net.eval()
        self.prop_net.eval()
        return self

