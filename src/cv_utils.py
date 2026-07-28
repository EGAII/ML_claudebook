"""Small helpers for your own experiments outside the notebooks.

The course notebooks are deliberately self-contained (they must run on a fresh Colab
VM that has never seen this repo), so they re-define the few functions they need.
This module is the tidied-up version of those helpers, for when you start writing
real scripts.

Usage from a notebook that *has* cloned the repo:

    import sys; sys.path.append('/content/ML_claudebook/src')
    from cv_utils import set_seed, get_device, show_images
"""

from __future__ import annotations

import os
import random
import time
from contextlib import contextmanager

import numpy as np

try:  # torch is optional so chapters 1 and part of 2 work without it
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


# --------------------------------------------------------------------------------------
# reproducibility & device
# --------------------------------------------------------------------------------------
def set_seed(seed: int = 0, deterministic: bool = False) -> None:
    """Seed python, numpy and torch.

    `deterministic=True` also forces cuDNN into deterministic mode. That makes runs
    bit-for-bit repeatable but can cost 10-30% throughput, so it is off by default;
    turn it on when you are chasing a bug, not when you are training.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_device(verbose: bool = True):
    """Return cuda if available, else mps (Apple), else cpu."""
    if not _HAS_TORCH:
        raise RuntimeError("torch is not installed")
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        dev = torch.device("mps")
        name = "Apple MPS"
    else:
        dev = torch.device("cpu")
        name = "CPU"
    if verbose:
        print(f"device: {dev} ({name})")
    return dev


@contextmanager
def timer(label: str = "block"):
    """`with timer('epoch'): ...` -> prints elapsed seconds."""
    t0 = time.perf_counter()
    yield
    print(f"{label}: {time.perf_counter() - t0:.3f}s")


# --------------------------------------------------------------------------------------
# tensor <-> image plumbing
# --------------------------------------------------------------------------------------
def to_hwc(img) -> np.ndarray:
    """Accept CHW tensor / CHW array / HWC array and return a HWC numpy array.

    Matplotlib wants HWC (or HW); torch wants CHW. Nearly every "why is my image
    garbage" bug is this conversion done wrong, so keep it in one place.
    """
    if _HAS_TORCH and isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    img = np.asarray(img)
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]
    return img


def denormalize(img, mean, std):
    """Undo `Normalize(mean, std)` so a tensor can be displayed.

    mean/std are per-channel sequences; img is CHW (tensor or array).
    """
    mean = np.asarray(mean).reshape(-1, 1, 1)
    std = np.asarray(std).reshape(-1, 1, 1)
    if _HAS_TORCH and isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    return np.clip(np.asarray(img) * std + mean, 0.0, 1.0)


def show_images(images, titles=None, ncols=4, figsize=None, cmap="gray", suptitle=None):
    """Plot a list of images in a grid. Accepts tensors or arrays, CHW or HWC."""
    import matplotlib.pyplot as plt

    images = list(images)
    n = len(images)
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols
    if figsize is None:
        figsize = (3.0 * ncols, 3.0 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if i >= n:
            continue
        ax.imshow(to_hwc(images[i]), cmap=cmap)
        if titles is not None and i < len(titles):
            ax.set_title(str(titles[i]), fontsize=10)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def plot_history(history: dict, figsize=(11, 4)):
    """Plot the {'train_loss': [...], 'val_loss': [...], 'val_acc': [...]} dict that
    the training loops in this course return. Loss left, metric right."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    for key in ("train_loss", "val_loss"):
        if key in history:
            ax1.plot(history[key], label=key, marker="o", ms=3)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend()
    ax1.grid(alpha=0.3)
    for key in history:
        if key.endswith(("acc", "iou", "dice", "f1")):
            ax2.plot(history[key], label=key, marker="o", ms=3)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("metric")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# metrics (numpy, framework independent)
# --------------------------------------------------------------------------------------
def accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    return float((y_true == y_pred).mean())


def confusion_matrix(y_true, y_pred, num_classes: int) -> np.ndarray:
    """Rows = true class, cols = predicted class. Built with bincount, so it is O(n)
    and works on millions of pixels (which is what you need for segmentation)."""
    y_true = np.asarray(y_true).ravel().astype(np.int64)
    y_pred = np.asarray(y_pred).ravel().astype(np.int64)
    keep = (y_true >= 0) & (y_true < num_classes)
    idx = y_true[keep] * num_classes + y_pred[keep]
    return np.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes)


def per_class_iou(cm: np.ndarray) -> np.ndarray:
    """IoU per class from a confusion matrix: TP / (TP + FP + FN)."""
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(denom > 0, tp / denom, np.nan)
    return iou


def mean_iou(y_true, y_pred, num_classes: int) -> float:
    """mIoU, ignoring classes that are absent from both truth and prediction."""
    iou = per_class_iou(confusion_matrix(y_true, y_pred, num_classes))
    return float(np.nanmean(iou))


def dice_score(y_true, y_pred, num_classes: int = 2) -> float:
    """Mean Dice (F1) over classes. 2*TP / (2*TP + FP + FN)."""
    cm = confusion_matrix(y_true, y_pred, num_classes)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        dice = np.where(denom > 0, 2 * tp / denom, np.nan)
    return float(np.nanmean(dice))


# --------------------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------------------
def count_parameters(model, trainable_only: bool = True) -> int:
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def conv_out_size(in_size: int, kernel: int, stride: int = 1, padding: int = 0, dilation: int = 1) -> int:
    """The formula you will use every time you design a conv stack:

        out = floor((in + 2*pad - dilation*(kernel-1) - 1) / stride) + 1
    """
    return (in_size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


if __name__ == "__main__":
    # tiny self-check: python src/cv_utils.py
    set_seed(0)
    yt = np.array([0, 0, 1, 1, 2, 2])
    yp = np.array([0, 1, 1, 1, 2, 0])
    print("accuracy  ", accuracy(yt, yp))
    print("confusion \n", confusion_matrix(yt, yp, 3))
    print("mIoU      ", round(mean_iou(yt, yp, 3), 4))
    print("dice      ", round(dice_score(yt, yp, 3), 4))
    print("conv_out  ", conv_out_size(32, 3, stride=2, padding=1))
