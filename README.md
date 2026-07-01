# SANER: Understanding SAM's Robustness to Noisy Labels through Gradient Down-weighting

Official code for the AISTATS 2026 paper **"Understanding SAM's Robustness to
Noisy Labels through Gradient Down-weighting"** by Hoang-Chau Luong*,
Quang-Thuc Nguyen*, Dat Ba Tran, and Minh-Triet Tran (*equal contribution).

[Paper (arXiv)](https://arxiv.org/abs/2411.17132) · AISTATS 2026, PMLR vol. 300

## Overview

Sharpness-Aware Minimization (SAM) generalizes well in the presence of label
noise, but it was not fully understood why. We show, both theoretically (in a
linear model) and empirically (in deep networks), that SAM **down-weights
element-wise gradients aligned with noisy supervision**: when a noisy gradient
dominates a parameter direction, SAM amplifies the clean gradient in that
direction more strongly than the noisy one, slowing memorization of the wrong
label while preserving clean learning.

Building on this, we propose **SANER** (Sharpness-Aware Noise-Explicit
Reweighting), a one-line modification of SAM that explicitly magnifies this
down-weighting effect:

```
r = g_SAM / g_SGD                      (element-wise gradient ratio)
m = 1[0 < r < 1]                       (down-weighted-element mask)
g_SANER = (1 - alpha * m) * g_SAM      (alpha is the only new hyperparameter)
```

SANER costs nothing extra over SAM (no additional gradient evaluations),
consistently improves test accuracy over SAM/SGD under symmetric, asymmetric,
instance-dependent, and real-world (CIFAR-N) label noise, and can be dropped
into other SAM-family optimizers (ASAM, GSAM, FSAM, VaSSO) to improve their
robustness too. See the paper for the full theory, ablations (model width,
dataset size, SAM's $\rho$), and comparisons with noise-robust training
pipelines (Co-teaching, DivideMix, PLC, Bootstrap).

## Repository structure

```
.
├── train.py                  # main entry point: SGD / SAM / SANER / SAM-family optimizers
├── baselines/                # entry points for noise-robust training pipelines
│   ├── train_bootstrap.py            # hard bootstrapping (Reed et al., 2014)
│   ├── train_co_teaching.py          # Co-teaching (Han et al., 2018)
│   ├── train_dividemix.py            # DivideMix (Li et al., 2020)
│   ├── train_plc.py                  # Progressive Label Correction
│   ├── train_joint_optimization.py   # Joint Optimization framework (Tanaka et al., 2018), stage 1
│   └── retrain_joint_optimization.py # Joint Optimization framework, stage 2 (retrain on corrected labels)
├── configs/                  # YAML training configs, one per optimizer/pipeline (see below)
├── scripts/                  # ready-to-run shell scripts that reproduce paper results
│   ├── main/                         # main CIFAR-10 SGD/SAM/SANER comparison (Fig. 2–4)
│   ├── baselines/                    # SANER + Bootstrap / Co-teaching / DivideMix / PLC
│   ├── ablation/                     # cross-architecture ablations (Appendix C.3)
│   └── miniwebvision/                # Mini-WebVision training
├── optimizer/                # SAM, SANER, and SAM-family optimizer implementations
├── models/                   # ResNet, WideResNet, DenseNet, SqueezeNet, InceptionResNetV2, ...
├── dataloader/                # CIFAR-10/100, Tiny-ImageNet, Mini-WebVision, Animal10N + noise injection
├── utils/                    # training loop, logging, PyHessian, early stopping, etc.
├── data/                     # dataset download/preparation scripts and instructions
├── toy_example/              # synthetic binary-classification illustration of SANER (Sec. 3)
└── visualize/                # plotting scripts + raw results behind every figure/table
```

## Installation

Requires Python ≥ 3.9 and a CUDA-capable GPU (experiments in the paper used a
single NVIDIA RTX 3090, 24GB).

```bash
git clone https://github.com/lhchau/sam-label-noise.git
cd sam-label-noise
pip install -r requirements.txt
```

Logging defaults to [Weights & Biases](https://wandb.ai/) (`wandb login`);
set `logging.framework_name: tensorboard` in a config (or
`--logging.framework_name=tensorboard` on the command line) to log to
TensorBoard under `runs/` instead.

## Datasets

CIFAR-10/100 download automatically on first use. Tiny-ImageNet, Mini-WebVision,
and the real-world CIFAR-N label files need a one-time manual download/prep —
see [`data/README.md`](data/README.md) for exact instructions.

## Quick start

Training is driven by a YAML config plus `--section.key=value` overrides
(a small "poor man's configurator", see `utils/configurator.py`). All commands
are run from the repository root.

```bash
# SGD baseline, CIFAR-100, 25% symmetric label noise, ResNet18
python train.py configs/sgd.yaml \
  --dataloader.data_name=cifar100 --dataloader.noise=0.25 \
  --model.model_name=resnet18 --logging.framework_name=wandb

# SAM baseline (same setting)
python train.py configs/sam.yaml \
  --dataloader.data_name=cifar100 --dataloader.noise=0.25 \
  --model.model_name=resnet18

# SANER (ours): SAM + explicit gradient down-weighting (alpha), linearly warmed
# up from 0 to its target value over the first `alpha_scheduler` epochs
python train.py configs/sam.yaml \
  --optimizer.opt_name=saner --optimizer.alpha=0.5 --trainer.alpha_scheduler=50 \
  --dataloader.data_name=cifar100 --dataloader.noise=0.25 \
  --model.model_name=resnet18
```

`configs/sam.yaml` and `configs/sgd.yaml` are the general-purpose templates
used for the main CIFAR-10/100 results in Tables 1–3 and Figures 1–4; the
optimizer, model, dataset, and noise setting are all selected via CLI
overrides as shown above.

### Optimizers (`--optimizer.opt_name=...`)

| `opt_name` | Optimizer | Implementation |
|---|---|---|
| `sgd` | SGD (+ momentum) | `torch.optim.SGD` |
| `sam` | SAM (Foret et al., 2021); set `--optimizer.adaptive=True` for ASAM | `optimizer/sam.py` |
| `saner` | **SANER (ours)** | `optimizer/saner.py` |
| `gsam` / `gsaner` | GSAM (Zhuang et al., 2022) / + SANER | `optimizer/gsam.py`, `optimizer/gsaner.py` |
| `fsam` / `fsaner` | Friendly-SAM (Li et al., 2024) / + SANER | `optimizer/fsam.py`, `optimizer/fsaner.py` |
| `vasso` / `vassosaner` | VaSSO (Li & Giannakis, 2024) / + SANER | `optimizer/vasso.py`, `optimizer/vassosaner.py` |
| `samonly`, `samwo`, `sanerlast` | Additional SAM ablation variants used in supplementary experiments | `optimizer/samonly.py`, `optimizer/samwo.py`, `optimizer/saner_last.py` |

Key hyperparameters: `rho` (SAM perturbation radius, default `0.1`), `alpha`
(SANER down-weighting strength, paper default `0.5`), `trainer.alpha_scheduler`
(linear warm-up length for `alpha`, paper default `50` epochs).

### Models (`--model.model_name=...`)

`resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`, `resnet32`,
`resnet56`, `wideresnet28_10`, `wideresnet40_2`, `densenet121`, `densenet169`,
`squeezenet`, `efficientnet_b0`, `inceptionresnetv2`, `resnet18_webvision`,
`resnet50_webvision` (the last two for Mini-WebVision). Use
`--model.widen_factor=<k>` to scale a ResNet's width (Appendix C.2).

### Label noise (`--dataloader...`)

```
--dataloader.data_name=cifar10|cifar100|tiny_imagenet|miniwebvision|animal10n
--dataloader.noise=0.25            # noise rate (fraction of flipped labels)
--dataloader.noise_type=symmetric|asymmetric|dependent|real
```

`dependent` is instance-dependent (PDN) noise; `real` reads the human-annotated
CIFAR-N labels described in [`data/README.md`](data/README.md). For CIFAR-N
and the Joint-Optimization/DivideMix/PLC pipelines (which need example
indices), use the `*_index` or `*_jo` dataset variants, e.g.
`--dataloader.data_name=cifar10_index`.

## Reproducing the paper

`scripts/` contains the exact commands used for every table and figure:

- **`scripts/main/reproduce_fig2_3_4_cifar10.sh`** — SGD vs. SAM vs.
  SGD-with-grouped-gradients, WideResNet28-10 on CIFAR-10 (Figures 2–4).
- **`scripts/ablation/`** — SANER vs. SAM/SGD across DenseNet121, ResNet34,
  and WideResNet40-2 on CIFAR-10/100 (Appendix C.2/C.3).
- **`scripts/miniwebvision/`** — SGD/SAM/SANER on Mini-WebVision with
  ResNet18 and InceptionResNetV2 (Table 2).
- **`scripts/baselines/`** — SANER combined with Bootstrap, Co-teaching,
  DivideMix, and PLC (Table 7 and related noise-robust-pipeline experiments).
  `co_teaching_quickstart.sh` is a minimal 2-command example; the rest are the
  full sweeps (multiple seeds/settings) used for the reported numbers.

Each `.sh` file is a flat list of `python train.py ...` (or
`python baselines/train_*.py ...`) invocations and can be run directly with
`bash scripts/<...>.sh`, or used as a template for new sweeps.

### Figures and toy example

- [`visualize/README.md`](visualize/README.md) maps every plotting script in
  `visualize/` to the figure/table it produces, along with the underlying
  result CSVs.
- [`toy_example/README.md`](toy_example/README.md) reproduces the synthetic
  binary-classification illustration (loss landscapes, decision boundaries,
  gradient-ratio contour plots) behind Section 3's theoretical analysis.

## Noise-robust baselines

In addition to the optimizer comparison, this repo includes from-scratch
implementations of several noise-robust *training pipelines*, used to show
that SANER is complementary to (rather than a replacement for) these methods:

| Pipeline | Entry point | Config(s) |
|---|---|---|
| Hard bootstrapping (Reed et al., 2014) | `baselines/train_bootstrap.py` | `configs/sam.yaml` |
| Co-teaching (Han et al., 2018) | `baselines/train_co_teaching.py` | `configs/sam_co_teaching.yaml`, `configs/sgd_co_teaching.yaml` |
| DivideMix (Li et al., 2020) | `baselines/train_dividemix.py` | `configs/sam_dividemix.yaml`, `configs/sgd_dividemix.yaml` |
| Progressive Label Correction | `baselines/train_plc.py` | `configs/sam_plc.yaml`, `configs/sgd_plc.yaml` |
| Joint Optimization (Tanaka et al., 2018) | `baselines/train_joint_optimization.py` then `baselines/retrain_joint_optimization.py` | `configs/sam.yaml` / `configs/sgd.yaml` |

The setup boilerplate shared by `train.py` and every `baselines/train_*.py`
script (seeding, run naming, logging init, dataloader/model/scheduler
construction) lives in `utils/experiment.py`; each script only implements
its own epoch loop / correction logic on top of it. The bootstrapping
baseline's loss is `utils/losses.py:HardBootstrappingLoss`.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{luong2026saner,
  title     = {Understanding {SAM}'s Robustness to Noisy Labels through Gradient Down-weighting},
  author    = {Luong, Hoang-Chau and Nguyen, Quang-Thuc and Tran, Dat Ba and Tran, Minh-Triet},
  booktitle = {Proceedings of the 29th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {300},
  year      = {2026},
  address   = {Tangier, Morocco},
  publisher = {PMLR}
}
```

(see also [`CITATION.bib`](CITATION.bib))

## License

Released under the [MIT License](LICENSE).

## Acknowledgments

Built on the official [SAM](https://github.com/davda54/sam) implementation
and draws on public implementations of ASAM, GSAM, Friendly-SAM, VaSSO,
Co-teaching, DivideMix, and the Joint Optimization framework. See the paper
for the complete reference list.
