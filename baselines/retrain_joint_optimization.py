"""Joint Optimization framework (Tanaka et al., 2018), stage 2: reloads the
corrected soft labels written by train_joint_optimization.py (stage 1) and
retrains from scratch against them with a soft cross-entropy loss."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, so `models`/`utils`/`dataloader`/`optimizer` resolve when run from this subdirectory

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import *
from utils import *
from dataloader import *
from optimizer import *

################################
#### 0. SETUP CONFIGURATION
################################
cfg = load_config()
device = get_device()
best_acc, start_epoch, logging_dict = 0, 0, {}

EPOCHS = cfg['trainer']['epochs']
resume = cfg['trainer'].get('resume', None)
alpha_scheduler = cfg['trainer'].get('alpha_scheduler', None)

logging_name, framework_name, writer = setup_experiment(cfg, tag=f'_k={alpha_scheduler}')

################################
#### 1. BUILD THE DATASET
################################
train_dataloader, val_dataloader, test_dataloader, num_classes = build_dataloaders(cfg)
train_dataloader.dataset.reload_label()  # load the soft labels written by stage 1

################################
#### 2. BUILD THE NEURAL NETWORK
################################
net = build_model(cfg, num_classes, device)

################################
#### 3.a OPTIMIZING MODEL PARAMETERS
################################
def retrain_criterion(outputs, soft_targets):
    return -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * soft_targets, dim=1))


test_criterion = nn.CrossEntropyLoss()
opt_name = cfg['optimizer'].pop('opt_name', None)
optimizer = get_optimizer(net, opt_name, cfg['optimizer'])
scheduler = build_scheduler(cfg['trainer'].get('scheduler', None), optimizer, EPOCHS)

################################
#### 3.b Training
################################
if __name__ == "__main__":
    if resume:
        for epoch in range(1, start_epoch + 1):
            scheduler.step()
    for epoch in range(start_epoch + 1, EPOCHS + 1):
        print('\nEpoch: %d' % epoch)
        if alpha_scheduler:
            optimizer.set_alpha(get_alpha(
                epoch, initial_alpha=1, final_alpha=cfg['optimizer']['alpha'], total_epochs=alpha_scheduler))

        loop_one_epoch_jo(
            dataloader=train_dataloader,
            net=net,
            criterion=retrain_criterion,
            optimizer=optimizer,
            device=device,
            logging_dict=logging_dict,
            epoch=epoch,
            loop_type='retrain',
            logging_name=logging_name)
        best_acc, acc = loop_one_epoch_jo(
            dataloader=test_dataloader,
            net=net,
            criterion=test_criterion,
            optimizer=optimizer,
            device=device,
            logging_dict=logging_dict,
            epoch=epoch,
            loop_type='test',
            logging_name=logging_name,
            best_acc=best_acc)
        if scheduler is not None:
            scheduler.step()

        log_metrics(framework_name, writer, logging_dict, epoch)
