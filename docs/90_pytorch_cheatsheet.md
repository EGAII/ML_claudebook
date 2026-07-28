# PyTorch cheatsheet — the API you actually use

Skim it now, come back to it constantly.

---

## Shapes and conventions

| Thing | Shape | dtype |
|---|---|---|
| Image batch | `(N, C, H, W)` | `float32`, normalized |
| Classification logits | `(N, num_classes)` | `float32` |
| Classification target | `(N,)` | **`int64`** |
| Segmentation logits | `(N, C, H, W)` | `float32` |
| Segmentation target | `(N, H, W)` | **`int64`** |
| Binary logits | `(N, 1)` | `float32` |
| Binary target | `(N, 1)` | `float32` |

## NumPy → PyTorch translation

| NumPy | PyTorch |
|---|---|
| `axis=` | `dim=` |
| `np.concatenate` | `torch.cat` |
| `np.stack` | `torch.stack` |
| `x[None]` | `x.unsqueeze(0)` (both work) |
| `x.astype(np.float32)` | `x.float()` |
| `np.transpose(x, (0,2,3,1))` | `x.permute(0,2,3,1)` |
| `x.reshape(n,-1)` | `x.reshape(n,-1)` or `x.view(n,-1)` (contiguous only) |
| `np.clip` | `torch.clamp` |
| `x.argmax(axis=1)` | `x.argmax(dim=1)` |
| `np.where(c,a,b)` | `torch.where(c,a,b)` |
| `x.mean(axis=(0,2,3))` | `x.mean(dim=(0,2,3))` |

```python
t = torch.from_numpy(arr)     # SHARES memory (CPU)
a = t.numpy()                 # SHARES memory (CPU only)
t.item()                      # scalar tensor -> python float
t.tolist()                    # tensor -> nested lists
t.detach()                    # same data, cut out of the autograd graph
t.cpu(), t.to(device)         # move between devices
```

## Layers

```python
nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
nn.ConvTranspose2d(c_in, c_out, kernel_size=2, stride=2)
nn.Linear(in_features, out_features)
nn.BatchNorm2d(num_features)         # normalize per channel over the batch
nn.LayerNorm(shape)                  # per sample; used in transformers
nn.GroupNorm(groups, channels)       # batch-size independent
nn.ReLU(inplace=True); nn.GELU(); nn.SiLU(); nn.LeakyReLU(0.1)
nn.MaxPool2d(2); nn.AvgPool2d(2); nn.AdaptiveAvgPool2d(1)   # GAP
nn.Dropout(p=0.5); nn.Dropout2d(p=0.1)
nn.Flatten()
nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
nn.Sequential(...); nn.ModuleList([...]); nn.ModuleDict({...})
```

**Output size:** `out = floor((in + 2*padding - dilation*(kernel-1) - 1) / stride) + 1`

**Params in a conv:** `c_out * (c_in // groups) * k * k + (c_out if bias else 0)`

Note `MaxPool2d`'s `stride` defaults to `kernel_size`, while `Conv2d`'s defaults to 1.

## Losses

```python
nn.CrossEntropyLoss()                      # LOGITS (N,C) + int64 target (N,)
nn.CrossEntropyLoss(weight=w,              # w: (C,) class weights
                    ignore_index=255,      # skip these target values
                    label_smoothing=0.1)
nn.BCEWithLogitsLoss()                     # LOGITS (N,1) + float target (N,1)
nn.MSELoss(); nn.L1Loss(); nn.SmoothL1Loss()
nn.NLLLoss()                               # needs log_softmax applied yourself
```

**Never apply `softmax`/`sigmoid` before these.** `CrossEntropyLoss` = `log_softmax` + `nll_loss`;
`BCEWithLogitsLoss` = `sigmoid` + `bce`, fused and numerically stable.

## Optimizers and schedules

```python
torch.optim.SGD(params, lr=0.05, momentum=0.9, nesterov=True, weight_decay=5e-4)
torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-2)
torch.optim.Adam(params, lr=1e-3)

torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)      # .step() per epoch
torch.optim.lr_scheduler.OneCycleLR(opt, max_lr, total_steps=...)  # .step() per BATCH
torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3)        # .step(val_metric)
torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
```

Parameter groups (different LR / no weight decay on norms and biases):

```python
decay, no_decay = [], []
for _, p in model.named_parameters():
    (no_decay if p.ndim <= 1 else decay).append(p)
opt = torch.optim.SGD([{'params': decay, 'weight_decay': 5e-4},
                       {'params': no_decay, 'weight_decay': 0.0}], lr=0.05, momentum=0.9)
```

## The training loop

```python
for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)      # 1
        logits = model(xb)                         # 2
        loss = criterion(logits, yb)               # 3
        loss.backward()                            # 4
        optimizer.step()                           # 5
    scheduler.step()

    model.eval()
    with torch.no_grad():
        ...
```

With mixed precision:

```python
scaler = torch.amp.GradScaler('cuda')
with torch.autocast('cuda'):
    loss = criterion(model(xb), yb)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Data

```python
class MyDataset(torch.utils.data.Dataset):
    def __len__(self): ...
    def __getitem__(self, i): return image_tensor, int64_label

DataLoader(ds, batch_size=128, shuffle=True, num_workers=2,
           pin_memory=True, drop_last=True, persistent_workers=True)

torch.utils.data.TensorDataset(X, y)
torch.utils.data.Subset(ds, indices)
torch.utils.data.random_split(ds, [800, 200])
torchvision.datasets.ImageFolder(root, transform)     # folder-per-class
```

## Transforms

```python
from torchvision import transforms as T          # or: from torchvision.transforms import v2 as T

T.Compose([T.RandomResizedCrop(224), T.RandomHorizontalFlip(),
           T.ColorJitter(0.2, 0.2, 0.2), T.ToTensor(), T.Normalize(mean, std)])
T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean, std)])
```

Augmentation **train only**; normalization **identical on both**. Order: PIL ops → `ToTensor` →
`Normalize`. ImageNet stats: `mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`.

`transforms.v2` transforms image + mask + boxes together — use it for segmentation and detection.

## Pretrained models

```python
from torchvision import models
w = models.ResNet18_Weights.IMAGENET1K_V1
model = models.resnet18(weights=w)
preprocess = w.transforms()                 # the exact preprocessing for these weights
classes = w.meta['categories']

model.fc = nn.Linear(model.fc.in_features, n)                 # resnet
model.classifier[-1] = nn.Linear(in_features, n)              # mobilenet / efficientnet
for p in model.parameters(): p.requires_grad = False          # freeze (do this BEFORE replacing)
```

## Save / load

```python
torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': ep}, path)
ckpt = torch.load(path, map_location=device)
model.load_state_dict(ckpt['model'])
model.load_state_dict(sd, strict=False)      # tolerate missing/extra keys
```

Save the `state_dict`, not the model object. Keep the **best** by validation metric.

## Inspection

```python
sum(p.numel() for p in model.parameters())
sum(p.numel() for p in model.parameters() if p.requires_grad)
[n for n, _ in model.named_parameters()]
[n for n, _ in model.named_children()]
print(model)

h = layer.register_forward_hook(lambda m, i, o: print(o.shape))
h.remove()
layer.register_full_backward_hook(lambda m, gi, go: ...)
```

## Reproducibility

```python
random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
torch.backends.cudnn.benchmark = True         # faster, fixed input shapes
torch.backends.cudnn.deterministic = True     # reproducible, ~10-30% slower
```

## Useful functionals

```python
import torch.nn.functional as F
F.conv2d(x, weight, bias, stride, padding)
F.interpolate(x, size=..., scale_factor=..., mode='bilinear'|'nearest')   # masks: nearest!
F.pad(x, (l, r, t, b), mode='constant'|'reflect'|'replicate')
F.one_hot(idx, num_classes)                  # (N,...) -> (N,...,C)
F.softmax(x, dim=1); F.log_softmax(x, dim=1)
F.normalize(x, dim=1)                        # L2 normalize
F.unfold(x, kernel_size)                     # im2col
torch.einsum('nchw,nchw->n', a, b)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## Quick numbers

| Quantity | Value |
|---|---|
| Initial loss, `C` balanced classes | `ln(C)` — 2.303 for 10, 0.693 for 2 |
| Receptive field, `n` stacked 3×3 stride-1 | `1 + 2n` |
| `Conv2d(3, 64, 7)` params | 9,472 |
| Dice from IoU | `2*IoU / (1 + IoU)` |
| float16 max | 65,504 (why `GradScaler` exists) |
| Typical batch size | 32–256; double batch ⇒ roughly double LR |
| GAN equilibrium `d_loss` | `2 ln 2` = 1.386, with D accuracy ≈ 0.5 |
| GAN optimizer | `Adam(2e-4, betas=(0.5, 0.999))`, init `N(0, 0.02)` |
| Diffusion loss floor | ≪ 1.0 (predicting zeros scores exactly `Var(eps)` = 1.0) |
| `‖z‖` for `z ~ N(0, I_d)` | ≈ `sqrt(d)` — so slerp, don't lerp |
| Guidance scale | 1 = true conditional; 3–8 typical; >10 saturates |

## Generative quick reference

```python
# --- GAN losses (ch 7)
d_loss = bce(D(real), ones) + bce(D(fake.detach()), zeros)   # detach!
g_loss = bce(D(fake), ones)                                  # non-saturating

# --- diffusion forward process (ch 8)
xt = sqrt_ab[t].view(-1,1,1,1) * x0 + sqrt_1mab[t].view(-1,1,1,1) * eps
loss = F.mse_loss(model(xt, t), eps)                         # t random PER IMAGE

# --- DDPM reverse step
x = (x - beta[t] / sqrt_1mab[t] * eps) / alpha[t].sqrt()
if t > 0:
    x = x + beta[t].sqrt() * torch.randn_like(x)             # no noise at t == 0

# --- classifier-free guidance
eps = eps_uncond + w * (eps_cond - eps_uncond)
```
