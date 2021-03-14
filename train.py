import time
from os import path
import datetime
import math

import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.multiprocessing as mp
import torch.distributed as distributed

from model.fusion_model import FusionModel
from dataset.fusion_dataset import FusionDataset

from util.logger import TensorboardLogger
from util.hyper_para import HyperParameters
from util.load_subset import *


"""
Initial setup
"""
torch.backends.cudnn.benchmark = True

# Init distributed environment
distributed.init_process_group(backend="nccl")
# Set seed to ensure the same initialization
torch.manual_seed(14159265)
np.random.seed(14159265)
random.seed(14159265)

print('CUDA Device count: ', torch.cuda.device_count())

# Parse command line arguments
para = HyperParameters()
para.parse()

local_rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
torch.cuda.set_device(local_rank)

print('I am rank %d in the world of %d!' % (local_rank, world_size))

"""
Model related
"""
if local_rank == 0:
    # Logging
    if para['id'].lower() != 'null':
        long_id = '%s_%s' % (datetime.datetime.now().strftime('%b%d_%H.%M.%S'), para['id'])
    else:
        long_id = None
    logger = TensorboardLogger(para['id'], long_id)
    logger.log_string('hyperpara', str(para))

    # Construct rank 0 model
    model = FusionModel(para, logger=logger, 
                    save_path=path.join('saves', long_id, long_id) if long_id is not None else None, 
                    local_rank=local_rank, world_size=world_size).train()
else:
    # Construct models of other ranks
    model = FusionModel(para, local_rank=local_rank, world_size=world_size).train()

# Load pertrained model if needed
if para['load_model'] is not None:
    total_iter = model.load_model(para['load_model'])
else:
    total_iter = 0

if para['load_network'] is not None:
    model.load_network(para['load_network'])

"""
Dataloader related
"""
if para['load_prop'] is None:
    print('Fusion module can only be trained with a pre-trained propagation module!')
    print('Use --load_prop [model_path]')
    raise NotImplementedError
model.load_prop(para['load_prop'])
torch.cuda.empty_cache()

if para['stage'] == 0:
    data_root = path.join(path.expanduser(para['bl_root']))
    train_dataset = FusionDataset(path.join(data_root, 'JPEGImages'), 
                    path.join(data_root, 'Annotations'), para['fusion_bl_root'])
elif para['stage'] == 1:
    data_root = path.join(path.expanduser(para['davis_root']), '2017', 'trainval')
    train_dataset = FusionDataset(path.join(data_root, 'JPEGImages', '480p'), 
                    path.join(data_root, 'Annotations', '480p'), para['fusion_root'])

                     
def worker_init_fn(worker_id): 
    return np.random.seed((torch.initial_seed()%2**31) + worker_id + local_rank*100)
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, rank=local_rank, shuffle=True)
train_loader = DataLoader(train_dataset, para['batch_size'], sampler=train_sampler, num_workers=8,
                            worker_init_fn=worker_init_fn, drop_last=True, pin_memory=True)

"""
Determine current/max epoch
"""
start_epoch = total_iter//len(train_loader)
total_epoch = math.ceil(para['iterations']/len(train_loader))
print('Actual training epoch: ', total_epoch)

"""
Starts training
"""
# Need this to select random bases in different workers
np.random.seed(np.random.randint(2**30-1) + local_rank*100)
try:
    for e in range(start_epoch, total_epoch): 
        # Crucial for randomness!
        train_sampler.set_epoch(e)

        # Train loop
        model.train()
        for data in train_loader:
            model.do_pass(data, total_iter)
            total_iter += 1

            if total_iter >= para['iterations']:
                break
finally:
    if not para['debug'] and model.logger is not None and total_iter > 5000:
        model.save(total_iter)
    # Clean up
    distributed.destroy_process_group()

