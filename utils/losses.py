import torch
import torch.nn as nn
import torch.nn.functional as F


class HardBootstrappingLoss(nn.Module):
    """Hard bootstrapping loss (Reed et al., "Training Deep Neural Networks
    on Noisy Labels with Bootstrapping", 2014), used by
    baselines/train_bootstrap.py.

    The cross-entropy target is a convex combination of the given (possibly
    noisy) label and the model's own current hard prediction, so the model
    is not forced to keep fitting a label it is already confident is wrong:

        L = -sum_k [beta * t_k + (1 - beta) * z_k] * log(q_k)

    where q is the softmax output, t is the one-hot given label, and z is
    the one-hot argmax of q (detached — treated as a fixed pseudo-label each
    step). `beta` close to 1 recovers standard cross-entropy; beta=0.8 is
    the value commonly used in CIFAR-scale reproductions and is the default
    here, matching the "Bootstrap (BS)" baseline referenced in Table 7 of
    the paper.
    """

    def __init__(self, beta: float = 0.8):
        super().__init__()
        self.beta = beta

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        num_classes = logits.size(1)

        target_onehot = F.one_hot(targets, num_classes=num_classes).float()
        with torch.no_grad():
            pseudo_onehot = F.one_hot(log_probs.argmax(dim=1), num_classes=num_classes).float()

        bootstrapped_target = self.beta * target_onehot + (1.0 - self.beta) * pseudo_onehot
        return -(bootstrapped_target * log_probs).sum(dim=1).mean()
