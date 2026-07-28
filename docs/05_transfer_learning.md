# Chapter 5 — Transfer learning: don't start from scratch

**Goal:** get a better result than chapter 4 in a fraction of the time and data, by reusing
features someone else paid to learn.

> **GPU runtime required** (Runtime → Change runtime type → T4 GPU).

Notebook: [`notebooks/05_transfer_learning.ipynb`](../notebooks/05_transfer_learning.ipynb) ·
Exercise: [`exercises/ex05_transfer.ipynb`](../exercises/ex05_transfer.ipynb)

---

## 1. Why it works

A network trained on ImageNet (1.2M images, 1000 classes) learns a **hierarchy**:

| Depth | What the filters respond to | How transferable |
|---|---|---|
| Layer 1 | oriented edges, colour blobs | universal — every vision task needs these |
| Early | corners, textures, simple patterns | very high |
| Middle | object parts — wheels, eyes, fur | high, within natural images |
| Late | whole-object templates | task-specific |
| Classifier | the 1000 ImageNet classes | useless to you — replace it |

Early layers are nearly universal because *any* natural image is made of edges and textures. So
you throw away only the last layer and keep the rest. Practically:

- **Data:** hundreds of images instead of hundreds of thousands.
- **Time:** minutes instead of days.
- **Accuracy:** usually *higher* than training from scratch on a small dataset — the pretrained
  features are better than anything your 500 images could teach.

Transfer learning is the default for real projects. Training from scratch is what you do when
your domain is genuinely unlike natural images (medical volumes, satellite multispectral,
audio spectrograms) — and even then, ImageNet initialization often still helps.

## 2. Two strategies

### Feature extraction (freeze the backbone)

Freeze everything, replace and train only the classifier head.

```python
model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
for p in model.parameters():
    p.requires_grad = False                      # freeze all
model.fc = nn.Linear(model.fc.in_features, n_classes)   # a NEW layer: requires_grad=True by default
```

- Fast (no backward pass through the backbone), tiny memory footprint.
- Very hard to overfit — only ~5k trainable parameters.
- Ceiling is lower: the features are fixed, so if your domain differs from ImageNet you can't adapt.

### Fine-tuning (train everything, gently)

Replace the head, then train the whole network with a **small** learning rate.

```python
model.fc = nn.Linear(model.fc.in_features, n_classes)    # nothing frozen
optimizer = torch.optim.SGD([
    {'params': backbone_params, 'lr': 1e-3},             # gentle: don't destroy good features
    {'params': model.fc.parameters(), 'lr': 1e-2},       # 10x: this layer is random
], momentum=0.9, weight_decay=1e-4)
```

The **discriminative learning rate** (10× lower for the backbone) matters. The head is randomly
initialized and produces large, meaningless gradients at first; at a uniform learning rate those
gradients propagate back and wreck the pretrained features in the first few steps. This failure
is called **catastrophic forgetting**, and it looks like "fine-tuning did worse than freezing".

A robust recipe when in doubt:

1. Freeze the backbone; train the head for 2–3 epochs (lets the head stop being random).
2. Unfreeze; train everything with a low LR and a cosine schedule.

### Which to pick

| Situation | Strategy |
|---|---|
| < ~1k images, similar to ImageNet | freeze the backbone |
| a few thousand images, similar domain | fine-tune the last block(s) |
| > ~10k images, or a different domain | fine-tune everything, low LR |
| tiny dataset, very different domain | freeze early layers, fine-tune late ones |

## 3. The preprocessing must match the weights

**This is the most common transfer learning bug.** Pretrained weights were fitted with a
specific normalization; feed them anything else and every activation is off-distribution.

```python
transforms.Resize(256)
transforms.CenterCrop(224)                        # ImageNet models expect 224x224
transforms.ToTensor()
transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet statistics - NOT your dataset's
                     std=[0.229, 0.224, 0.225])
```

Use the **ImageNet** statistics, not statistics computed from your own data. The weights encode
an expectation about the input distribution, and matching it is more important than being
optimal for your data.

Modern torchvision gives you the right transform automatically, which removes the whole class of
error:

```python
weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
model = torchvision.models.resnet18(weights=weights)
preprocess = weights.transforms()      # exactly the preprocessing these weights were trained with
```

Grayscale input? ImageNet models want 3 channels. Either `img.convert('RGB')` (repeats the
channel) or replace the first conv and sum its weights across the input dimension.

## 4. What to replace, per architecture

The classifier layer has a different name in each family:

| Model | Head attribute | Replace with |
|---|---|---|
| ResNet, ShuffleNet | `model.fc` | `nn.Linear(model.fc.in_features, n)` |
| VGG, AlexNet | `model.classifier[6]` | `nn.Linear(4096, n)` |
| DenseNet | `model.classifier` | `nn.Linear(model.classifier.in_features, n)` |
| EfficientNet, MobileNetV3 | `model.classifier[-1]` | `nn.Linear(in_features, n)` |
| ViT | `model.heads.head` | `nn.Linear(in_features, n)` |

Always read `in_features` off the existing layer rather than hard-coding 512 — it changes between
model sizes, and a hard-coded number is a silent bug when you swap resnet18 for resnet50.

`print(model)` is the fastest way to find the head of an unfamiliar model.

## 5. BatchNorm needs care when freezing

Freezing with `requires_grad = False` stops *gradient* updates but **not** BatchNorm's running
statistics — those update in `train()` mode regardless, because they're buffers, not parameters.
With a small dataset that quietly drifts your frozen features away from what the weights expect.

To truly freeze a backbone, also put its norm layers in eval mode:

```python
def freeze_bn(module):
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

model.train()          # every epoch, after this...
freeze_bn(model.layer1)  # ...re-apply, because model.train() resets it
```

This is a real, commonly-missed subtlety. If frozen-backbone results drift epoch to epoch for no
apparent reason, this is why.

## 6. Interpreting the model: Grad-CAM

Was the model right for the right reason? **Grad-CAM** produces a heatmap of which spatial
locations drove a prediction:

1. Forward to get logits; pick the target class score.
2. Backprop to the last conv layer's feature maps $A^k$.
3. Weight each channel by its mean gradient: $\alpha_k = \overline{\partial y_c / \partial A^k}$.
4. Heatmap $= \text{ReLU}\left(\sum_k \alpha_k A^k\right)$, upsampled to the input.

Intuition: the gradient says "how much would increasing this feature map raise the class score",
so a gradient-weighted sum of the maps is "where the evidence was".

Implemented with **forward and backward hooks** — worth learning for their own sake, since
they're how you inspect any intermediate value in PyTorch:

```python
feats, grads = {}, {}
layer.register_forward_hook(lambda m, i, o: feats.__setitem__('a', o.detach()))
layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__('g', go[0].detach()))
```

Grad-CAM regularly reveals that a classifier is using the *background* (grass ⇒ dog, snow ⇒
wolf). That's a dataset problem no amount of tuning fixes, and you'd never see it from the loss.

## 7. Practical notes

- **Freeze → train head → unfreeze** is a safe default that rarely loses to anything fancier.
- **Smaller LR for a smaller dataset.** With 500 images, `1e-4` on the backbone.
- **Augment more, not less.** Small datasets overfit fast; augmentation is your cheapest defence.
- **`model.eval()` at inference**, always — pretrained models are full of BatchNorm.
- **Watch the first epoch.** If validation accuracy starts high and then *drops*, your backbone LR
  is too high and you're forgetting the pretrained features.
- **Weights download once** per Colab session into `~/.cache/torch/hub`; wiped on disconnect.

## 8. What this buys you, concretely

On a ~400-image, 2-class dataset:

| Approach | Val accuracy | Training time |
|---|---|---|
| Small CNN from scratch | ~0.70 | minutes |
| ResNet18, frozen backbone | ~0.93 | ~30 seconds |
| ResNet18, fine-tuned | ~0.96 | ~1 minute |

That gap doesn't close by tuning the from-scratch model. It closes by not throwing away 1.2
million labelled images someone else already paid for.

---

## Checklist before moving on

- [ ] Which normalization statistics for a pretrained model, and why?
- [ ] How do you freeze a backbone, and what does freezing *not* freeze?
- [ ] Why give the new head a 10× larger learning rate?
- [ ] What is catastrophic forgetting, and what does it look like in your curves?
- [ ] How do you find the classifier layer of an unfamiliar model?
- [ ] What does Grad-CAM actually compute?
- [ ] 300 images of a domain unlike ImageNet — freeze or fine-tune?

Next: [Chapter 6 — Semantic segmentation](06_segmentation.md)
