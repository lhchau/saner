"""Co-teaching (Han et al., 2018) combined with SGD/SAM/SANER: two networks
trained jointly, each selecting the other's low-loss ("clean-looking")
samples to learn from."""
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
#### 2. BUILD THE NEURAL NETWORKS (co-teaching trains a pair)
################################
net1 = build_model(cfg, num_classes, device)
net2 = build_model(cfg, num_classes, device)

################################
#### 3.a OPTIMIZING MODEL PARAMETERS
################################
criterion = nn.CrossEntropyLoss()
criterion_vec = nn.CrossEntropyLoss(reduction="none")  # per-sample loss, used for co-teaching's small-loss selection

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
    use_coteaching = cfg['trainer'].get('coteaching', True)
    forget_rate = cfg['trainer'].get('forget_rate', 0.2)
    num_gradual = cfg['trainer'].get('num_gradual', 10)
    exponent = cfg['trainer'].get('exponent', 1.0)

    for epoch in range(1, EPOCHS + 1):
        print('\nEpoch: %d' % epoch)
        if alpha_scheduler:
            optimizer1.set_alpha(get_alpha(
                epoch, initial_alpha=1, final_alpha=cfg['optimizer']['alpha'], total_epochs=alpha_scheduler))
            optimizer2.set_alpha(get_alpha(
                epoch, initial_alpha=1, final_alpha=cfg['optimizer']['alpha'], total_epochs=alpha_scheduler))

        loop_one_epoch_co_teaching(
            dataloader=train_dataloader,
            net=(net1, net2) if use_coteaching else net1,
            criterion=criterion,
            optimizer=(optimizer1, optimizer2) if use_coteaching else optimizer1,
            device=device,
            logging_dict=logging_dict,
            epoch=epoch,
            loop_type='train',
            logging_name=logging_name,
            coteaching=use_coteaching,
            criterion_vec=criterion_vec,
            forget_rate=forget_rate,
            num_gradual=num_gradual,
            exponent=exponent,
            total_epochs=EPOCHS,
        )

        best_acc, acc = loop_one_epoch(
            dataloader=test_dataloader,
            net=net1,              # evaluate net1
            criterion=criterion,
            optimizer=optimizer1,  # unused in test
            device=device,
            logging_dict=logging_dict,
            epoch=epoch,
            loop_type='test',
            logging_name=logging_name,
            best_acc=best_acc)

        scheduler1.step()
        scheduler2.step()

        log_metrics(framework_name, writer, logging_dict, epoch)
