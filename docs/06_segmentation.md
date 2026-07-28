# Chapter 6 — Semantic segmentation: from *what* to *where*

**Goal:** predict a class for **every pixel**. Build a U-Net from scratch, use the right loss and
the right metric, and know the handful of pitfalls that make segmentation code silently wrong.

> **GPU runtime required** (Runtime → Change runtime type → T4 GPU).

Notebook: [`notebooks/06_segmentation_unet.ipynb`](../notebooks/06_segmentation_unet.ipynb) ·
Exercise: [`exercises/ex06_segmentation.ipynb`](../exercises/ex06_segmentation.ipynb)

---

## 1. The task, and how it differs

| Task | Output | Shape |
|---|---|---|
| Classification | one label per image | `(N, C)` |
| Object detection | boxes + labels | variable |
| **Semantic** segmentation | class per pixel, instances merged | `(N, C, H, W)` |
| **Instance** segmentation | class per pixel, instances separate | per-object masks |
| **Panoptic** | semantic + instance combined | both |

Semantic segmentation is per-pixel classification. That single sentence tells you almost
everything:

- **Loss:** cross-entropy — the same one. Just applied `H×W` times per image.
- **Prediction:** `logits.argmax(dim=1)`, taking `(N, C, H, W)` → `(N, H, W)`. The same call as
  chapter 2, one rank higher.
- **Labels:** an integer mask `(N, H, W)` of dtype `int64`, **not** one-hot.

`nn.CrossEntropyLoss` handles the spatial case natively — it expects input `(N, C, H, W)` and
target `(N, H, W)`. You don't reshape anything.

What genuinely changes: you need **output at input resolution**, which the downsampling
architecture of chapter 4 has destroyed. That's the architectural problem this chapter solves.

## 2. Why the classifier architecture doesn't work

Chapter 4's CNN goes 32×32 → 4×4 → global pool → 10 logits. Two things went wrong for
segmentation:

1. Spatial resolution is gone (4×4 can't label 1024 pixels meaningfully).
2. Global pooling deliberately discards position — exactly the information we now need.

There's a genuine tension:

- **Deep features** (large receptive field) know *what* something is, but are coarse.
- **Shallow features** (high resolution) know *where* edges are, but not what they belong to.

You need both. Every segmentation architecture is an answer to "how do I combine them".

## 3. The encoder–decoder, and U-Net

**Encoder** (a normal CNN): downsample, grow channels, build semantic features.
**Decoder**: upsample back to full resolution, shrinking channels.
**Skip connections**: concatenate the encoder's high-resolution features into the matching decoder
stage.

```
input 128x128 ──[enc1]── 64ch ─────────────skip─────────────► [dec1] ── out 128x128
                  │                                              ▲
                pool                                          upsample
                  ▼                                              │
              [enc2]── 128ch ──────────skip──────────────► [dec2]
                  │                                          ▲
                pool                                      upsample
                  ▼                                          │
              [bottleneck] 256ch ──────────────────────────────
```

The skip connections **are** U-Net's contribution, and they matter enormously: without them a
decoder must reconstruct boundary detail from a coarse map, and predictions come out blobby.
Diagnosing "my masks are blurry and boundaries are wrong" almost always ends at the skips.

They also give gradients a short path to early layers, which makes the whole thing easier to
train — the same reason ResNet's residual connections work.

### Upsampling: two options

```python
# Option A: learned transposed convolution
nn.ConvTranspose2d(c_in, c_out, kernel_size=2, stride=2)

# Option B: fixed interpolation + a normal conv (usually preferred)
nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
nn.Conv2d(c_in, c_out, 3, padding=1)
```

`ConvTranspose2d` learns the upsampling but produces **checkerboard artifacts** when the kernel
size isn't divisible by the stride. Option B avoids them and often works better. Use `2×2 stride 2`
if you do use transposed conv — the one case where the arithmetic is clean.

**Never use bilinear interpolation on a label mask.** Bilinear on a mask containing classes 1 and 3
produces 2.4, which is class 2 — a class that isn't there. Masks resize with **nearest neighbour,
always.**

## 4. Metrics: accuracy is actively misleading

If 95% of pixels are background, "predict background everywhere" scores 95% pixel accuracy while
finding nothing. Segmentation therefore uses overlap metrics.

**IoU** (Jaccard), per class:

$$\text{IoU} = \frac{|P \cap G|}{|P \cup G|} = \frac{TP}{TP + FP + FN}$$

**Dice** (F1 on pixels):

$$\text{Dice} = \frac{2|P \cap G|}{|P| + |G|} = \frac{2TP}{2TP + FP + FN}$$

Both are 0 (no overlap) to 1 (perfect); note **neither counts true negatives**, which is exactly
why they don't reward predicting background. They're monotonically related
($\text{Dice} = 2\text{IoU}/(1+\text{IoU})$), so they rank models identically — Dice is just
always the more flattering number, which is worth knowing when you read a paper.

**mIoU** — mean IoU over classes — is the standard headline metric. Two details that matter:

- Average over **classes**, not pixels, or big classes dominate again.
- A class absent from both prediction and ground truth has IoU 0/0. Report `NaN` and use
  `np.nanmean`, or your score depends on which images happened to be in the batch.

Compute all of this from a confusion matrix built with `np.bincount` — one pass over millions of
pixels (chapter 1's trick, now earning its keep).

## 5. Losses

**Cross-entropy** — the default. Per-pixel, well-behaved gradients, works.

```python
criterion = nn.CrossEntropyLoss()             # input (N,C,H,W) logits, target (N,H,W) int64
criterion = nn.CrossEntropyLoss(weight=w)     # w: (C,) tensor, upweight rare classes
criterion = nn.CrossEntropyLoss(ignore_index=255)   # skip "unlabelled" pixels entirely
```

`ignore_index` is essential on real datasets, where boundary pixels are often marked unlabelled
rather than forced into a class.

**Dice loss** — optimize the metric directly, in a differentiable form:

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2\sum p g + \epsilon}{\sum p + \sum g + \epsilon}$$

using **softmax probabilities** $p$ (not argmax — that has no gradient) and one-hot $g$. Naturally
handles class imbalance, because it's a ratio rather than a sum over pixels. But its gradients are
noisier, and it degenerates when a class is absent from a batch (hence $\epsilon$).

**The practical answer: use both.**

```python
loss = ce_loss(logits, target) + dice_loss(logits, target)
```

CE gives stable gradients everywhere; Dice pushes overlap directly. This combination is what most
competitive segmentation solutions use. **Focal loss** (down-weighting easy pixels) is the third
common ingredient, for extreme imbalance.

## 6. The pitfalls, all in one place

These are the bugs that produce code which runs, trains, and is wrong.

1. **Mask dtype.** Must be `int64`. A float mask gives a cryptic error from `CrossEntropyLoss`; a
   `uint8` mask can silently overflow.
2. **Mask interpolation.** Nearest neighbour only. Bilinear invents classes.
3. **Augmentation must transform image *and* mask identically.** Two independent `RandomCrop`
   calls give you an image and a mask of different regions, and the model learns nothing. Use
   `transforms.v2` (which handles pairs), or seed both calls identically, or write the geometry by
   hand.
4. **No softmax before `CrossEntropyLoss`.** Same rule as chapter 2, still the same bug.
5. **Class values must be `0..C-1`.** A dataset with classes `{0, 1, 255}` needs remapping, or
   `ignore_index=255`.
6. **Output size must equal input size.** Off-by-one from odd input sizes is common — pad the input
   to a multiple of your total downsampling factor (16 for a 4-level U-Net), or crop the skips.
7. **Don't report pixel accuracy.** Report per-class IoU and mIoU.
8. **Check the class balance before training** (`np.bincount(mask.ravel())`). It tells you whether
   you need weighting.

## 7. Real architectures, briefly

| Architecture | Idea |
|---|---|
| **FCN** (2015) | first end-to-end: replace the classifier with 1×1 conv, upsample |
| **U-Net** (2015) | symmetric encoder–decoder with concatenating skips. Still the default |
| **SegNet** | decoder reuses the encoder's max-pool indices |
| **DeepLab v3+** | dilated convolution + atrous spatial pyramid pooling: large receptive field without losing resolution |
| **PSPNet** | pyramid pooling for multi-scale context |
| **SegFormer / Mask2Former** | transformer-based; current state of the art |

In practice you'd use a **pretrained encoder** (chapter 5!) with a U-Net decoder — the
`segmentation_models_pytorch` library is one line:

```python
import segmentation_models_pytorch as smp
model = smp.Unet('resnet34', encoder_weights='imagenet', classes=3)
```

Writing one from scratch first is how you understand what that line is doing. torchvision also
ships pretrained `deeplabv3_resnet50` and `fcn_resnet50` for the 21 Pascal VOC classes.

---

## Checklist before moving on

- [ ] Shape and dtype of logits and target for segmentation cross-entropy?
- [ ] Why is pixel accuracy a bad metric? What do you use instead?
- [ ] What do skip connections do, and what's the symptom of not having them?
- [ ] Why must masks be resized with nearest neighbour?
- [ ] Difference between IoU and Dice? Why does neither count true negatives?
- [ ] What does `ignore_index` do, and when do you need it?
- [ ] Your augmentation crops the image and the mask separately. What happens?
- [ ] Input 100×100 into a 4-level U-Net. What goes wrong, and how do you fix it?

*(Answers: logits `(N,C,H,W)` float32, target `(N,H,W)` int64; because background dominates — use
per-class IoU and mIoU; they carry high-resolution detail to the decoder, without them masks are
blobby with bad boundaries; bilinear interpolation invents class values that don't exist; Dice
counts intersection twice, and neither includes TN so neither is rewarded for predicting
background; it excludes unlabelled pixels from the loss, needed whenever a dataset marks boundary
or void pixels; the mask no longer corresponds to the image and the model learns nothing; 100 isn't
divisible by 16 so the decoder output won't match the input size — pad to 112 or crop the skips.)*

---

## End of the discriminative half

Chapters 1–6 all had the same shape: an image goes in, a label or a mask comes out, and a ground-truth
answer tells you how wrong you were. You can now:

- Read a tensor shape and know what every axis means.
- Write a training loop from memory, and debug one that's silently broken.
- Explain convolution well enough to design a network on paper.
- Fine-tune a pretrained model correctly, and check *why* it's right.
- Build an encoder–decoder with skip connections and evaluate it honestly.

Chapters 7–8 remove the ground truth. **Generative** modelling asks for a *new* image, so there is
nothing to subtract your output from — and defining the objective becomes the entire problem. Two
very different answers to that:

- [**Chapter 7 — GANs**](07_gans.md): if you can't write the loss, *learn* it, as a second network.
- [**Chapter 8 — Diffusion**](08_diffusion.md): reframe generation as denoising, which makes the loss
  a plain MSE — and reuses the U-Net you just built as its backbone. That closing remark about
  diffusion was not a throwaway line; the architecture transfers almost unchanged.

Next: [Chapter 7 — GANs](07_gans.md)
