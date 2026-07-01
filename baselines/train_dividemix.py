"""DivideMix (Li et al., 2020) combined with SGD/SAM/SANER: two networks
trained jointly; after a supervised warmup, each network fits a per-sample
loss GMM to split the data into a clean/noisy partition for the *other*
network, which then trains semi-supervised on that split (MixMatch-style)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, so `models`/`utils`/`dataloader`/`optimizer` resolve when run from this subdirectory

import torch.nn as nn

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
alpha_scheduler = cfg['trainer'].get('alpha_scheduler', None)

logging_name, framework_name, writer = setup_experiment(cfg, tag=f'_k={alpha_scheduler}')

################################
#### 1. BUILD THE DATASET
################################
train_dataloader, val_dataloader, test_dataloader, num_classes = build_dataloaders(cfg)

################################
#### 2. BUILD THE NEURAL NETWORKS (DivideMix trains a co-training pair)
################################
net1 = build_model(cfg, num_classes, device)
net2 = build_model(cfg, num_classes, device)

################################
#### 3.a OPTIMIZING MODEL PARAMETERS
################################
criterion_sup = nn.CrossEntropyLoss()
criterion_vec = nn.CrossEntropyLoss(reduction="none")  # per-sample loss, used to fit the clean/noisy GMM

opt_name = cfg['optimizer'].pop('opt_name', None)
optimizer1 = get_optimizer(net1, opt_name, cfg['optimizer'])
optimizer2 = get_optimizer(net2, opt_name, cfg['optimizer'])
scheduler_name = cfg['trainer'].get('scheduler', None)
scheduler1 = build_scheduler(scheduler_name, optimizer1, EPOCHS)
scheduler2 = build_scheduler(scheduler_name, optimizer2, EPOCHS)

################################
#### 3.b Training
################################
if __name__ == "__main__":
    warmup_epochs = cfg['trainer'].get('warmup_epochs', 10)
    p_threshold = cfg['trainer'].get('p_threshold', 0.5)
    lambda_u = cfg['trainer'].get('lambda_u', 25.0)
    T = cfg['trainer'].get('T', 0.5)
    alpha = cfg['trainer'].get('alpha', 4.0)
    n_train = len(train_dataloader.dataset)

    for epoch in range(1, EPOCHS + 1):
        print('\nEpoch: %d' % epoch)

        if epoch <= warmup_epochs:
            # ---- warmup: standard supervised training, both nets ----
            loop_one_epoch_warmup(train_dataloader, net1, optimizer1, device, criterion_sup, logging_dict, epoch, logging_name, tag="net1")
            loop_one_epoch_warmup(train_dataloader, net2, optimizer2, device, criterion_sup, logging_dict, epoch, logging_name, tag="net2")
        else:
            # ---- 1) estimate each network's own clean-sample probabilities ----
            losses1 = eval_loss_per_sample(net1, train_dataloader, device, criterion_vec, n_samples=n_train)
            losses2 = eval_loss_per_sample(net2, train_dataloader, device, criterion_vec, n_samples=n_train)
            p_clean1 = fit_gmm_two_components(losses1)
            p_clean2 = fit_gmm_two_components(losses2)

            # ---- 2) train net1 using net2's clean/noisy split, and vice versa ----
            train_dividemix_epoch(
                dataloader=train_dataloader, net=net1, net_other=net2, optimizer=optimizer1,
                device=device, num_classes=num_classes, p_clean_other=p_clean2,
                p_threshold=p_threshold, lambda_u=lambda_u, T=T, alpha=alpha,
                logging_dict=logging_dict, epoch=epoch, logging_name=logging_name, tag="net1")

            train_dividemix_epoch(
                dataloader=train_dataloader, net=net2, net_other=net1, optimizer=optimizer2,
                device=device, num_classes=num_classes, p_clean_other=p_clean1,
                p_threshold=p_threshold, lambda_u=lambda_u, T=T, alpha=alpha,
                logging_dict=logging_dict, epoch=epoch, logging_name=logging_name, tag="net2")

            # ---- test net1 ----
            best_acc, acc = loop_one_epoch(
                dataloader=test_dataloader, net=net1, criterion=criterion_sup, optimizer=optimizer1,
                device=device, logging_dict=logging_dict, epoch=epoch,
                loop_type='test', logging_name=logging_name, best_acc=best_acc)

        scheduler1.step()
        scheduler2.step()

        log_metrics(framework_name, writer, logging_dict, epoch)
