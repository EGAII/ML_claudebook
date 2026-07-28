# Debugging playbook

Symptom → likely cause → what to do. Ordered by how often it happens.

The meta-rule first: **most deep learning bugs don't raise an exception.** The code runs, the loss
decreases, the model is quietly worse than it should be. So the fastest debugging strategy is
usually not reading the stack trace — it's the three checks below.

---

## The three checks, before anything else

```python
# 1. Look at a batch, AFTER transforms, with labels as titles
xb, yb = next(iter(train_loader))
plot(denormalize(xb[i]), title=CLASSES[yb[i]])

# 2. Initial loss should be ln(num_classes)
print(criterion(model(xb), yb).item(), 'vs expected', math.log(num_classes))

# 3. Overfit 8 samples to ~0 loss
xs, ys = xb[:8].to(device), yb[:8].to(device)
for _ in range(250):
    opt.zero_grad(); loss = criterion(model(xs), ys); loss.backward(); opt.step()
assert loss.item() < 0.01
```

Check 3 is the important one. It splits every problem into two disjoint worlds:

- **Can't reach ~0** → there is a **bug**. Stop tuning. Look at the loop, the labels, the gradient
  flow.
- **Reaches ~0 but validation is bad** → **not** a bug. That's regularization, data, or
  hyperparameters — i.e. the normal work.

Turn dropout off and augmentation off for this test; both keep the loss floor above zero.

---

## Loss is `nan` or `inf`

In rough order of likelihood:

1. **Learning rate too high.** Divide by 10 and retry. This is the answer more often than everything
   else combined.
2. **`log(0)` in a custom loss.** Clamp probabilities: `p.clamp(1e-7, 1 - 1e-7)`, or better, use the
   fused `*WithLogitsLoss`.
3. **Division by zero** — an empty class in a Dice/IoU loss. Add `eps` to numerator *and* denominator.
4. **Input already contains `nan`.** `assert torch.isfinite(xb).all()`. A `std` of 0 in your
   normalization does this.
5. **Exploding gradients** in a deep or recurrent model. `clip_grad_norm_(params, 1.0)`.
6. **fp16 overflow** under AMP. A sum over many elements exceeds 65,504 → `inf`. Cast that reduction
   to `.float()`.

Find the first bad step:

```python
torch.autograd.set_detect_anomaly(True)     # slow; pinpoints the offending op
```

---

## Loss doesn't move at all (flat at `ln(C)`)

| Cause | Check |
|---|---|
| Forgot `optimizer.step()` | read the loop |
| Forgot `loss.backward()` | `p.grad is None` for every parameter |
| Learning rate ~0 | `print(opt.param_groups[0]['lr'])` |
| Everything frozen | `sum(p.requires_grad for p in model.parameters())` |
| Optimizer built on the wrong parameters | built *before* you replaced the head? |
| Graph broken | a `.detach()`, `.item()`, or numpy round-trip mid-forward |
| Labels are random | plot a batch with its labels |
| `optimizer.zero_grad()` inside the wrong loop | gradients cleared after accumulating |

Diagnostic: gradient norms per layer. All-zero means no gradient is arriving.

```python
for n, p in model.named_parameters():
    print(n, None if p.grad is None else round(p.grad.norm().item(), 6))
```

---

## Loss decreases but accuracy stays at chance

- **Double softmax.** You applied `softmax` before `CrossEntropyLoss`. Doesn't crash; trains badly.
- **Target shape wrong.** `(N,1)` vs `(N,)` — broadcasting silently computes something else.
  Modern PyTorch warns; read the warning.
- **Class indices out of range**, or 1-based instead of 0-based.
- **Comparing wrongly.** `logits.argmax(1)` vs `(logits > 0)` — pick the one that matches your loss.
- **Labels shuffled relative to images.** `reset_index(drop=True)` after filtering a DataFrame, or a
  positional index into a filtered frame.

---

## Train loss falls, validation loss rises (overfitting)

In the order to try them:

1. **More data.** The only real fix.
2. **More augmentation.** Cheapest real fix.
3. **Weight decay** `1e-4` to `5e-4` (excluded from norms and biases).
4. **Early stopping** — keep the best checkpoint by validation metric.
5. **Dropout** before the classifier.
6. **Smaller model.**
7. **Transfer learning** — a pretrained backbone needs far less data.

---

## Both losses plateau high (underfitting)

- Learning rate too low, or too few epochs.
- Model too small, or too little receptive field for the object size.
- Inputs not normalized.
- Too much augmentation — check you haven't made the task impossible (a 0.3-scale crop of a 32×32
  image often is).
- Labels genuinely noisy: look at the confident mistakes and ask whether *you* could do the task.

---

## Validation accuracy jitters between epochs, or is worse than train for no reason

- **Forgot `model.eval()`.** BatchNorm then uses batch statistics, so your metric depends on batch
  composition. This is the number one cause.
- Validation set too small — a 100-image set has ±3% noise.
- Learning rate too high late in training. Use a cosine schedule.
- Batch size too small for BatchNorm (<8 is unreliable; use GroupNorm).

Validation *better* than train is usually **normal**: train is measured on augmented inputs with
dropout active, validation on clean inputs with dropout off.

---

## Shape errors

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x2048 and 512x10)
```
The flatten size doesn't match your `Linear`. Print shapes through the forward pass, or use
`AdaptiveAvgPool2d(1)` so the head is input-size agnostic.

```
Expected 4-dimensional input for 4-dimensional weight
```
Missing batch or channel dimension. `x[None]`, or `x[:, None]` for a grayscale mask.

```
Expected more than 1 value per channel when training
```
A batch of size 1 hit BatchNorm. Use `drop_last=True`, or `model.eval()` if you're only doing
inference.

```
Expected all tensors to be on the same device
```
Something wasn't `.to(device)`. Usually a loss weight tensor, or a tensor created inside `forward`.

```
view size is not compatible with input tensor's size and stride
```
Use `.reshape()` instead of `.view()` after a `permute`/`transpose`.

```
Trying to backward through the graph a second time
```
You called `.backward()` twice, or kept a tensor across iterations. Recompute the forward pass, or
`retain_graph=True` if you genuinely need it.

```
grad can implicitly be created only for scalar outputs
```
You forgot `.mean()` or `.sum()` on your loss.

```
Boolean value of Tensor with more than one element is ambiguous
```
An `if tensor:` or a `tensor in list`. Compare with `id()` or use `.any()` / `.all()`.

---

## CUDA out of memory

```python
torch.cuda.empty_cache()          # helps only with fragmentation
```

Real fixes, in order of value per effort:

1. **Smaller batch size.** Halve it.
2. **Mixed precision** (`autocast` + `GradScaler`) — roughly halves activation memory.
3. **`@torch.no_grad()` around evaluation.** Storing the graph during validation is a classic leak.
4. **Don't accumulate tensors.** `total += loss.item()`, not `total += loss` — the latter keeps every
   epoch's graph alive.
5. **Gradient accumulation** to emulate a big batch: step every `k` micro-batches.
6. **`torch.utils.checkpoint`** to trade compute for memory.
7. **Restart the runtime.** After an OOM the Python traceback holds references to the tensors that
   caused it; in Colab, *Runtime → Disconnect and delete runtime* is the reliable reset.

`nvidia-smi` or `torch.cuda.memory_allocated() / 1e9` to see where you are.

---

## Training is slow

| Fix | Typical gain |
|---|---|
| Mixed precision (AMP) | ~2× |
| `num_workers=2..4` in the DataLoader | large if you were CPU-bound |
| `pin_memory=True` + `non_blocking=True` | small but free |
| `torch.backends.cudnn.benchmark = True` | 5-20% at fixed input shapes |
| Bigger batch (if memory allows) | better GPU utilization |
| Don't call `.item()` / `.cpu()` every step | removes a GPU sync point |
| Copy the dataset to local disk, not Drive | huge on Colab |
| `torch.compile(model)` | 10-50%, PyTorch 2.x |

Diagnose first: if GPU utilization is low, you're **data-bound**, and a faster model won't help.

---

## Segmentation-specific

| Symptom | Cause |
|---|---|
| Masks blurry, boundaries wrong | skip connections missing or wired to the wrong stage |
| mIoU near 0 while pixel accuracy is 0.95 | model predicts only background — expected early; check the metric you're watching |
| Classes appear that aren't in your labels | mask resized with bilinear interpolation |
| Loss decreases, mIoU stays near 0 | image and mask augmented independently |
| `CrossEntropyLoss` rejects the target | mask isn't `int64`, or has a channel axis |
| Output a few pixels smaller than input | input size not divisible by the total downsampling factor |
| mIoU jumps wildly between epochs | averaging per-batch IoU instead of accumulating one confusion matrix |

---

## GAN specific

| Symptom | Cause |
|---|---|
| All samples nearly identical | mode collapse — lower G's LR, label smoothing, spectral norm, WGAN-GP |
| D loss → 0, G loss → ∞ | D won; **weaken** D (smaller, lower LR, smooth its labels) |
| Samples are pure noise, both losses flat | D too weak to teach; or `.detach()` missing so G is being trained by D's loss |
| G never updates at all | `zero_grad()` called after `backward()`; or `opt_G` built on D's parameters |
| Loss goes to `-inf` then `nan` | you used the negated saturating loss; use `bce(D(fake), ones)` |
| Samples have a visible grid pattern | checkerboard artifacts — kernel size not divisible by stride in `ConvTranspose2d` |
| D hits 100% accuracy in the first epoch | output activation doesn't match the data range (tanh vs `[0,1]` data) |
| Generator ignores the class label | D isn't conditioned on the label |
| Interpolation midpoints look washed out | linear interpolation in latent space — use slerp |
| "Loss went down, is it working?" | **GAN loss is not a progress signal.** Look at samples and D's accuracy |

## Diffusion specific

| Symptom | Cause |
|---|---|
| Loss plateaus at exactly 1.0 | the model is predicting zeros — target noise differs from the noise used in `q_sample` |
| Samples are structured garbage | `sqrt_recip_alphas` built from `alpha_bar` instead of `alpha`; check the reverse-step coefficients |
| All samples nearly identical | you forgot to inject noise during sampling (or added it on the final step too) |
| Final samples look grainy | noise added at `t == 0`; guard with `if t > 0` |
| `nan` immediately | `beta` reaching 1 (divide by `sqrt(alpha)=0`) — clamp betas to ≤ 0.999 |
| Broadcast error in `q_sample` | schedule table indexed `(B,)` needs `.view(-1, 1, 1, 1)` |
| Reconstruction check fails at all `t` | `sqrt_ab` / `sqrt_1mab` swapped |
| Check fine at small `t`, huge at large `t` | multiplying by `sqrt_ab` where you should divide |
| Class label has no effect | forgot the `+1` null slot, or the embedding isn't added to the time embedding |
| Guidance does nothing | `p_uncond=0`, so the null token was never trained |
| Samples over-smoothed at few steps | normal DDIM discretization error — use more steps or a better solver |

**The check that catches most of these:** verify that
`predict_x0_from_eps(q_sample(x0, t, eps), t, eps) == x0` at many `t`, before you train anything. It's
an exact algebraic identity, so any deviation beyond float32 noise is a bug.

## Transfer learning specific

| Symptom | Cause |
|---|---|
| Fine-tuning is worse than freezing | backbone LR too high → catastrophic forgetting |
| Val accuracy peaks at epoch 0 then falls | same |
| Accuracy mediocre for no clear reason | wrong normalization — use `weights.transforms()` |
| Frozen backbone gives different results each epoch | BatchNorm running stats still updating; also call `.eval()` on the BN layers |
| Shape error in the head | hard-coded `in_features`; read it from the layer |
| Grad-CAM heat on the background | the model learned a shortcut — a **data** problem |

---

## When you're properly stuck

1. **Minimise.** Cut to 10 samples, 1 epoch, no augmentation, batch size 2. Does it still happen?
2. **Bisect.** Revert to the last version that worked; re-apply changes one at a time.
3. **Compare against a reference.** Run a torchvision model on your data, or your model on
   CIFAR-10. Whichever works tells you which side the bug is on.
4. **Print shapes and dtypes everywhere.** Most silent bugs are a broadcast or a cast you didn't
   intend.
5. **Check a hand-written gradient numerically.** Four lines, catches everything.
6. **Look at the data again.** Not the loader — the actual pixels, with the actual labels. An
   astonishing fraction of "model bugs" are label bugs.
