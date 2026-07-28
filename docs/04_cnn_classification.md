# Chapter 4 — CNN image classification: the full training pipeline

**Goal:** train a real CNN on real images, end to end, with the habits that make the
difference between a model that works and one you can't debug.

> **Switch Colab to a GPU runtime before starting:** Runtime → Change runtime type → **T4 GPU**.
> On CPU this notebook takes ~45 min/epoch; on a T4, ~25 seconds.

Notebook: [`notebooks/04_cnn_classification.ipynb`](../notebooks/04_cnn_classification.ipynb) ·
Exercise: [`exercises/ex04_cnn.ipynb`](../exercises/ex04_cnn.ipynb)

---

## 1. The data pipeline: `Dataset` and `DataLoader`

Two objects with one job each.

**`Dataset`** answers "give me sample `i`". You need exactly three methods:

```python
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df, self.transform = df.reset_index(drop=True), transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row.path).convert('RGB')     # load ONE image
        if self.transform:
            img = self.transform(img)                 # -> tensor, CHW, normalized
        return img, int(row.class_id)                 # (tensor, int64 label)
```

That's the manifest `DataFrame` from chapter 1 finally paying off. Note what it does *not* do:
no batching, no shuffling, no parallelism. Those belong to the loader.

**`DataLoader`** turns single samples into batches:

```python
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2)
```

| Argument | Why |
|---|---|
| `shuffle=True` | **train only.** Without it, batches are correlated and the model sees the classes in order |
| `num_workers=2` | subprocesses that decode/augment while the GPU computes. On Colab, 2 is the sweet spot |
| `pin_memory=True` | faster host→GPU copies. Free when you have a GPU |
| `drop_last=True` | train only — drops the final partial batch, which keeps BatchNorm statistics stable |
| `persistent_workers=True` | avoids re-spawning workers every epoch (worth it for short epochs) |

**Never shuffle validation.** You want the same order every epoch so the numbers are comparable.

## 2. Transforms: preprocessing vs augmentation

The two pipelines are *different*, and mixing them up is a real bug:

```python
train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),      # augmentation (random)
    transforms.RandomHorizontalFlip(),         # augmentation (random)
    transforms.ToTensor(),                     # HWC uint8 0-255 -> CHW float 0-1
    transforms.Normalize(MEAN, STD),           # preprocessing
])

val_tf = transforms.Compose([
    transforms.ToTensor(),                     # NO augmentation
    transforms.Normalize(MEAN, STD),           # SAME preprocessing
])
```

Rules:

1. **Augmentation on train only.** Random transforms on validation make your metric noisy and
   pessimistic, and you can no longer tell a real improvement from luck.
2. **Identical preprocessing on both.** Different normalization between train and val is the
   silent killer: training looks fine, validation is garbage, and nothing errors.
3. **Order matters.** Geometric/PIL ops → `ToTensor()` → `Normalize`. `ToTensor` is the boundary
   between "PIL image" and "tensor".
4. **Augment to match reality.** Horizontal flip is free accuracy on natural photos and
   actively harmful on digits (a flipped 2 is not a 2). Think about what variation your
   *deployment* data actually has.

`torchvision.transforms.v2` is the current API — same idea, faster, and it transforms images
*and* masks/boxes together, which matters in chapter 6.

## 3. The architecture

Chapter 3's block, stacked:

```
[Conv-BN-ReLU] x2 -> MaxPool  32x32 ->16x16   16 ch
[Conv-BN-ReLU] x2 -> MaxPool  16x16 -> 8x8    32 ch
[Conv-BN-ReLU] x2 -> MaxPool   8x8  -> 4x4    64 ch
GlobalAvgPool -> Dropout -> Linear(64, 10)
```

Design rules that carry over to anything you build:

- Channels double when spatial size halves.
- 3×3 kernels, `padding=1`, `bias=False` before BatchNorm.
- End with global average pooling, not a giant flatten.
- Dropout right before the classifier only. Dropout between convolutions is mostly obsolete —
  BatchNorm already regularizes, and the two interact badly.

## 4. The training loop, properly

```python
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', enabled=scaler is not None):
            logits = model(xb)
            loss = criterion(logits, yb)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * yb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        seen += yb.size(0)
    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total_loss += criterion(logits, yb).item() * yb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        seen += yb.size(0)
    return total_loss / seen, correct / seen
```

Details that are not optional:

- **`model.train()` / `model.eval()`** — BatchNorm and Dropout behave differently. Getting this
  wrong gives noisy, pessimistic validation numbers.
- **`@torch.no_grad()`** on evaluation — no graph, so it's faster and uses much less memory.
- **`set_to_none=True`** — slightly faster than zeroing, and it's the default in recent PyTorch.
- **Weight the running loss by batch size** (`* yb.size(0)`) and divide by the sample count.
  Averaging batch means is wrong when the last batch is smaller.
- **`.item()` when accumulating** — otherwise you keep the graph alive and leak memory.
- **`non_blocking=True`** with `pin_memory=True` overlaps the host→device copy with compute.

### Mixed precision (AMP)

`torch.autocast` runs most ops in float16/bfloat16: roughly **2× faster** and half the memory on
a T4, at no measurable accuracy cost. `GradScaler` compensates for float16's small dynamic
range by scaling the loss up before `backward()` and unscaling before the step — without it,
small gradients flush to zero.

Use it for every real training run. It's four lines.

## 5. Optimizer and schedule

Sensible defaults, in the order to try them:

| Choice | Default | Note |
|---|---|---|
| Optimizer | `SGD(lr=0.05, momentum=0.9, nesterov=True, weight_decay=5e-4)` | best final accuracy on vision tasks |
| Alternative | `AdamW(lr=1e-3, weight_decay=1e-2)` | converges faster, less tuning, sometimes ~1% worse |
| Schedule | `CosineAnnealingLR(T_max=epochs)` | smooth decay to ~0; nearly always helps |
| Batch size | as large as fits, then tune LR | double batch → roughly double LR |

**Momentum** accumulates a velocity across steps, damping oscillation across the ravine and
accelerating along it. It's essentially free and always on.

**Weight decay on BatchNorm parameters and biases is a mistake** people make constantly. Decay
weights, not norms or biases:

```python
decay, no_decay = [], []
for name, p in model.named_parameters():
    (no_decay if p.ndim <= 1 else decay).append(p)
optimizer = torch.optim.SGD([{'params': decay, 'weight_decay': 5e-4},
                             {'params': no_decay, 'weight_decay': 0.0}], lr=0.05, momentum=0.9)
```

**Learning rate matters more than everything else combined.** Order of tuning: LR → epochs →
augmentation → architecture. Never start with architecture.

## 6. The habits that catch bugs

### Overfit one batch — do this first, always

Take 8 samples. Turn off augmentation. Train until the loss is ~0.

```python
xb, yb = next(iter(train_loader))
xb, yb = xb[:8].to(device), yb[:8].to(device)
for i in range(200):
    optimizer.zero_grad(); loss = criterion(model(xb), yb); loss.backward(); optimizer.step()
```

If it **can't** reach ~0, no hyperparameter will save you — there's a bug in your model,
loss, or label handling. If it **can**, your pipeline is wired correctly and everything after
is tuning. This is the single highest-value 30 seconds in deep learning.

### The rest of the checklist

1. **Look at a batch.** Plot the images *after* transforms, with their labels. Catches
   label misalignment, wrong normalization, broken augmentation.
2. **Check the initial loss.** For `C` balanced classes, an untrained model should give
   `ln(C)` — 2.303 for 10 classes. A wildly different value means broken initialization or
   labels.
3. **Plot train and val curves every run.** The shape diagnoses more than the number.
4. **Look at the confident mistakes.** They're often mislabeled data, not model failures.
5. **Seed everything**, and know that full determinism costs speed.
6. **Save checkpoints to Drive** — Colab disconnects and `/content` is wiped.

### Diagnosing from the curves

| Symptom | Likely cause |
|---|---|
| Loss `nan` in the first few steps | LR too high; or bad input normalization; or `log(0)` in a custom loss |
| Loss flat at `ln(C)` | LR far too low; labels shuffled; frozen parameters; forgot `optimizer.step()` |
| Train loss falls, val loss rises | overfitting → more augmentation, weight decay, early stop |
| Both plateau high | underfitting → bigger model, more epochs, higher LR |
| Val accuracy jitters wildly | forgot `model.eval()`; batch too small; LR too high late in training |
| Val *better* than train | normal with heavy augmentation + dropout (train is measured on harder inputs) |

## 7. Saving and loading

Save the `state_dict`, not the model object (pickling a class binds you to your source layout):

```python
torch.save({'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'best_acc': best_acc}, '/content/drive/MyDrive/ckpt/best.pt')

ckpt = torch.load(path, map_location=device)
model.load_state_dict(ckpt['model'])
```

Keep the **best** checkpoint by validation metric, not the last — that's early stopping without
having to stop early.

## 8. Evaluating properly

Accuracy is one number and hides everything. Also produce:

- **Confusion matrix**, row-normalized. Which classes get confused *with which*?
- **Per-class recall.** One class at 40% while the average is 85% is a data problem.
- **The most confident mistakes.** Highest predicted probability among wrong predictions.
- **A reliability check.** If the model says 90%, is it right 90% of the time? Modern networks
  are systematically **overconfident**.

Test-time augmentation (average predictions over an image and its mirror) usually buys ~0.5%
for 2× the inference cost. Cheap trick, good to know.

---

## Checklist before moving on

- [ ] Write `Dataset.__getitem__` from memory. What does it return?
- [ ] Which transforms go on train but not validation, and why?
- [ ] Expected initial loss for 10 balanced classes?
- [ ] Your loss is `nan` at step 3. What do you check first?
- [ ] Why weight the running loss by batch size?
- [ ] What does `GradScaler` do, and why is it needed?
- [ ] Why exclude BatchNorm parameters from weight decay?
- [ ] Your model can't reach ~0 loss on 8 samples. What does that tell you?

Next: [Chapter 5 — Transfer learning](05_transfer_learning.md)
