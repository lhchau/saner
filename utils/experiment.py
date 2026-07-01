"""
Shared experiment-setup helpers used by train.py and every script in
baselines/. Each training script differs only in its per-method epoch loop
(see loop_one_epoch.py / jo_regularization.py / the PLC correction pass,
etc.) — the boilerplate around it (seeding, run naming, logging framework
init, dataloader/model/scheduler construction) is identical and lives here.
"""
import os
import pprint
import datetime

import wandb
import torch
from torch.utils.tensorboard import SummaryWriter

from .utils import initialize, get_logging_name
from .configurator import exec_configurator

from models import get_model
from dataloader import get_dataloader


def load_config():
    """Parses the YAML config + `--section.key=value` CLI overrides (see
    configurator.py) into a config dict."""
    return exec_configurator()


def setup_experiment(cfg, tag=""):
    """Seeds all RNGs from `cfg['trainer']['seed']`, builds a unique run name
    (config summary + optional `tag`, e.g. the alpha-scheduler length + a
    timestamp), and initializes the configured logging framework.

    Must be called *before* anything pops keys out of `cfg['optimizer']`
    (e.g. `opt_name`), since `get_logging_name` summarizes the full config.

    Returns (logging_name, framework_name, writer). `writer` is a
    `SummaryWriter` when `framework_name == 'tensorboard'`, otherwise None
    (metrics are logged to Weights & Biases instead — see `log_metrics`).
    """
    initialize(cfg['trainer']['seed'])
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    print('==> Initialize Logging Framework..')
    logging_name = get_logging_name(cfg) + tag + '_' + current_time

    framework_name = cfg['logging']['framework_name']
    writer = None
    if framework_name == 'wandb':
        wandb.init(project=cfg['logging']['project_name'], name=logging_name, config=cfg)
    elif framework_name == 'tensorboard':
        tb_log_dir = os.path.join('runs', cfg['logging']['project_name'], logging_name)
        writer = SummaryWriter(log_dir=tb_log_dir)
    pprint.pprint(cfg)

    return logging_name, framework_name, writer


def log_metrics(framework_name, writer, logging_dict, epoch):
    """Sends one epoch's worth of metrics to whichever logging framework is
    active."""
    if framework_name == 'wandb':
        wandb.log(logging_dict)
    elif framework_name == 'tensorboard':
        for metric_name, metric_value in logging_dict.items():
            writer.add_scalar(metric_name, metric_value, epoch)


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def build_dataloaders(cfg):
    """Builds train/(val)/test dataloaders from `cfg['dataloader']`.
    `val_dataloader` is None unless `cfg['dataloader']['use_val']` is set."""
    if cfg['dataloader'].get('use_val', False):
        train_dataloader, val_dataloader, test_dataloader, num_classes = get_dataloader(**cfg['dataloader'])
    else:
        train_dataloader, test_dataloader, num_classes = get_dataloader(**cfg['dataloader'])
        val_dataloader = None
    return train_dataloader, val_dataloader, test_dataloader, num_classes


def build_model(cfg, num_classes, device):
    net = get_model(**cfg['model'], num_classes=num_classes).to(device)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"==> Number of parameters in {cfg['model']}: {total_params}")
    return net


def build_scheduler(scheduler_name, optimizer, epochs):
    """The three LR schedules used throughout the paper's experiments:
    cosine annealing, the two-milestone step schedule used for Tiny-ImageNet,
    and the default 50%/75%-of-training step schedule used everywhere else.
    """
    if scheduler_name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == 'tiny_imagenet':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80])
    else:
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[int(epochs * 0.5), int(epochs * 0.75)])
