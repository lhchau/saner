# Datasets

This project trains on CIFAR-10, CIFAR-100, Tiny-ImageNet, and Mini-WebVision under
four kinds of label noise: symmetric, asymmetric, instance-dependent (PDN), and
real-world human annotation noise (CIFAR-N). CIFAR-10/100 and Tiny-ImageNet are
downloaded automatically by `torchvision`/the data loaders the first time you train,
with one exception: the **real-world noise labels (CIFAR-N)** and **Mini-WebVision**
must be prepared manually, as described below.

## CIFAR-10 / CIFAR-100 (symmetric, asymmetric, instance-dependent noise)

No setup needed. `dataloader/cifar10.py` and `dataloader/cifar100.py` download the
clean datasets via `torchvision.datasets` on first use (default location: `./data`)
and synthesize noisy labels on the fly according to `--dataloader.noise` and
`--dataloader.noise_type` (`symmetric`, `asymmetric`, or `dependent` for
instance-dependent PDN noise).

## CIFAR-N (real-world human noise)

CIFAR-10/100 with real human-annotated label noise, from
[Wei et al., "Learning with Noisy Labels Revisited" (ICLR 2022)](https://github.com/UCSC-REAL/cifar-10-100n).
The data loaders read the noisy labels from `./data/CIFAR-10_human.pt` and
`./data/CIFAR-100_human.pt` whenever `--dataloader.noise_type=real` (or any other
data-loading path that reaches the "real-world noise" branch). Download both files
from the [CIFAR-N repository](https://github.com/UCSC-REAL/cifar-10-100n/tree/main/data)
and place them at:

```
data/CIFAR-10_human.pt
data/CIFAR-100_human.pt
```

We use the **"Worst"** label set for CIFAR-10N and the **"Fine"** label set for
CIFAR-100N, matching the paper (Appendix B).

## Tiny-ImageNet

`dataloader/tiny_imagenet.py` expects the standard Tiny-ImageNet-200 layout at
`./data/tiny-imagenet-200`. Download and unpack it with:

```bash
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip -P data/
unzip data/tiny-imagenet-200.zip -d data/
```

## Mini-WebVision

We follow the "Mini" protocol of Jiang et al. (2018): the first 50 classes of the
Google-image subset of [WebVision](https://data.vision.ee.ethz.ch/cvl/webvision/webvision_data.html),
evaluated against the matching 50 classes of the clean ImageNet-2012 validation set.
`data/miniwebvision/` contains the scripts used to build this dataset:

```bash
cd data/miniwebvision
bash prepare_miniwebvision.sh
```

This downloads the WebVision Google-resized images, the validation images, and the
WebVision `info` metadata, then reshuffles both splits into an ImageNet-style
`train/<class>/*.jpg` and `val/<class>/*.jpg` layout (using
`build_imagenet_folder_map.py` to map WebVision query folders to ImageNet class
names) so that it can be read with a standard `torchvision.datasets.ImageFolder`
via `dataloader/miniwebvision.py`. This requires roughly 30GB of free disk space
and a working `wget`.

After preparation, point `--dataloader.data_name=miniwebvision` at the resulting
directory (see `scripts/miniwebvision/`).
