# Chapter 2 — ML foundations: regression, gradient descent, autograd

**Goal:** understand the machinery every deep model shares, using the two simplest models
that have it. Linear regression and logistic regression are not warm-ups you outgrow — a
CNN classifier *is* logistic regression with a learned feature extractor bolted on the front.

Notebook: [`notebooks/02_linear_logistic_regression.ipynb`](../notebooks/02_linear_logistic_regression.ipynb) ·
Exercise: [`exercises/ex02_regression.ipynb`](../exercises/ex02_regression.ipynb)

---

## 1. The supervised learning contract

You have pairs $(x_i, y_i)$. You pick a **model family** $f_\theta$, a **loss** $L$ that
scores how wrong a prediction is, and then you search for parameters:

$$\theta^* = \arg\min_\theta \frac{1}{N}\sum_{i=1}^{N} L\big(f_\theta(x_i),\, y_i\big)$$

Four decisions, and every one of them is yours to make:

| Decision | Linear regression | Logistic regression | CNN classifier |
|---|---|---|---|
| Model | $w^\top x + b$ | $\sigma(w^\top x + b)$ | conv stack → linear |
| Loss | MSE | binary cross-entropy | cross-entropy |
| Optimizer | GD / closed form | GD | SGD / Adam |
| Output | a real number | a probability | class probabilities |

The training loop is identical in all three columns. That's the point of this chapter.

## 2. Linear regression

**Model.** With features stacked into a design matrix $X \in \mathbb{R}^{N \times D}$:

$$\hat{y} = Xw + b$$

**Loss.** Mean squared error:

$$L(w, b) = \frac{1}{N}\sum_i (\hat{y}_i - y_i)^2$$

**Gradients.** Write $r = \hat{y} - y$ (the residual). Then:

$$\frac{\partial L}{\partial w} = \frac{2}{N} X^\top r \qquad
\frac{\partial L}{\partial b} = \frac{2}{N} \sum_i r_i$$

Derive this once by hand and you will recognise the shape of every gradient you ever meet:
*(upstream error) × (local derivative)*, summed over the batch. That is backpropagation in
one line.

Shape check — the gradient of a parameter always has the parameter's shape:
$X^\top r$ is $(D, N) \times (N,) = (D,)$, same as $w$. If your gradient's shape doesn't
match your parameter's, you have a bug, not a subtlety.

**Closed form.** MSE with a linear model is one of the very few cases with an exact solution
(the *normal equation*):

$$w = (X^\top X)^{-1} X^\top y$$

Use `np.linalg.lstsq`, never literally `inv(X.T @ X)` — the explicit inverse is numerically
unstable when features are correlated. The closed form matters here as *ground truth*: it
tells you the lowest loss gradient descent could possibly reach, so you can tell "my
optimizer is broken" apart from "my model can't fit this".

Once you add a nonlinearity, the closed form is gone forever and iteration is all you have.

## 3. Gradient descent

$$\theta \leftarrow \theta - \eta \nabla_\theta L$$

$\eta$ is the learning rate, and it is the hyperparameter you will spend the most time on:

| $\eta$ | Symptom |
|---|---|
| Too small | loss decreases, agonisingly; you conclude the model can't learn |
| Good | smooth decrease, flattening out |
| Slightly too large | loss zig-zags but trends down |
| Too large | loss increases, then `inf`, then `nan` |

**If your loss is NaN, suspect the learning rate first.** It is the answer more often than
anything else.

Three flavours, differing only in how much data one step sees:

- **Batch GD** — all $N$ samples per step. Smooth, slow, needs everything in memory.
- **Stochastic GD** — one sample per step. Noisy; the noise is a mild regularizer.
- **Mini-batch GD** — 32–256 samples. What everyone actually uses: enough samples to
  saturate the GPU, enough noise to escape bad spots.

One **epoch** = one full pass over the training set = `ceil(N / batch_size)` steps.

### Feature scaling is not optional

If feature 1 is in $[0, 1]$ and feature 2 is in $[0, 10000]$, the loss surface is a long
thin valley. A learning rate small enough to be stable along the steep axis is far too small
to make progress along the flat one. Standardize:

$$x' = \frac{x - \mu}{\sigma}$$

using **training-set** statistics, applied unchanged to validation and test. This is exactly
what `transforms.Normalize` does for images, and why we computed per-channel stats in
chapter 1.

## 4. PyTorch: tensors and autograd

A `torch.Tensor` is a NumPy array that can (a) live on a GPU and (b) remember how it was
computed.

```python
x = torch.tensor([[1., 2.], [3., 4.]])   # from data
torch.zeros(3, 4); torch.randn(2, 3); torch.arange(6).reshape(2, 3)
t = torch.from_numpy(arr)                # SHARES memory with the numpy array
a = t.numpy()                            # also shares - CPU only
t.to('cuda'); t.cpu(); t.float(); t.long()
t.shape, t.dtype, t.device
```

Naming differences from NumPy worth memorising:

| NumPy | PyTorch |
|---|---|
| `axis=` | `dim=` |
| `reshape` | `reshape` (or `view`, contiguous-only) |
| `concatenate` | `cat` |
| `img[None]` | `unsqueeze(0)` (or `img[None]`, both work) |
| `x.astype(np.float32)` | `x.float()` |
| `np.transpose(x,(1,0))` | `x.permute(1, 0)` / `x.T` |
| `x @ y` | `x @ y` (same) |

### Autograd in four rules

```python
w = torch.tensor([1.0, 2.0], requires_grad=True)   # 1. mark it as a parameter
loss = ((X @ w - y) ** 2).mean()                   # 2. forward: a graph is recorded
loss.backward()                                    # 3. backward: fills w.grad
with torch.no_grad():                               # 4. update outside the graph
    w -= lr * w.grad
w.grad.zero_()                                      #    and clear the gradient
```

1. **`requires_grad=True`** makes a tensor a leaf of the computation graph. Everything
   downstream of it is tracked.
2. **`.backward()`** walks the graph in reverse, accumulating $\partial \text{loss} / \partial \theta$
   into each leaf's `.grad`. It works on a **scalar** — for a non-scalar you must supply a
   gradient, which is nearly always a sign you forgot `.mean()`.
3. **Gradients accumulate.** They add to whatever is already in `.grad`. This is a feature
   (it's how you emulate a large batch on a small GPU), but forgetting `zero_grad()` is the
   most common training bug in existence: your updates use the sum of all gradients since
   the beginning, so the loss wobbles and never converges.
4. **`torch.no_grad()`** disables tracking. Use it for the parameter update, for validation,
   and for inference. It also saves a lot of memory.

`loss.item()` pulls a Python float out of a scalar tensor. Accumulating `loss` itself
instead of `loss.item()` keeps the entire graph alive across the epoch and will OOM you.

`.detach()` returns a tensor sharing the same data but cut out of the graph. Use it when you
want a value, not a gradient path.

## 5. The canonical training loop

Memorise this. Everything else in the course is this loop with a different model.

```python
model = nn.Linear(D, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(n_epochs):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()          # 1. clear old gradients
        pred = model(xb)               # 2. forward
        loss = criterion(pred, yb)     # 3. loss
        loss.backward()                # 4. backward
        optimizer.step()               # 5. update

    model.eval()
    with torch.no_grad():
        ...                            # validate
```

- `model.train()` / `model.eval()` only matter when the model contains dropout or batch
  norm — but by then you'll have forgotten to add them, so write them from day one.
  Forgetting `eval()` is the classic "validation accuracy is mysteriously worse than
  training and jitters between runs".
- `optimizer.zero_grad()` at the **top** of the step, not the bottom. Same effect, but it
  survives `continue` statements and early exits.
- The order 1-2-3-4-5 never changes.

## 6. Logistic regression: classification

We want a probability, so squash the linear score with the **sigmoid**:

$$\sigma(z) = \frac{1}{1 + e^{-z}} \in (0, 1), \qquad z = w^\top x + b$$

$z$ is called the **logit**. Loss is **binary cross-entropy**:

$$L = -\frac{1}{N}\sum_i \Big[ y_i \log \hat{p}_i + (1 - y_i)\log(1 - \hat{p}_i) \Big]$$

Why not MSE? Two reasons: BCE is the negative log-likelihood of a Bernoulli model (so it's
the *right* loss under the model you're assuming), and its gradient doesn't vanish when the
model is confidently wrong. MSE + sigmoid gives you a gradient of nearly zero exactly when
you most need a large correction.

And the beautiful part — the gradient is *identical in form* to linear regression:

$$\frac{\partial L}{\partial w} = \frac{1}{N} X^\top (\hat{p} - y)$$

Same "$X^\top \times$ residual". That is not a coincidence; it's a property of this whole
family of models.

### Never apply sigmoid yourself before the loss

```python
# WRONG - numerically fragile
p = torch.sigmoid(logits)
loss = nn.BCELoss()(p, y)

# RIGHT - one fused, stable op
loss = nn.BCEWithLogitsLoss()(logits, y)
```

`log(sigmoid(z))` overflows for large $|z|$; the fused version uses the log-sum-exp trick
internally. The same applies to multi-class:

```python
# nn.CrossEntropyLoss expects RAW LOGITS and applies log_softmax internally
loss = nn.CrossEntropyLoss()(logits, target)   # logits (N, C) float, target (N,) int64
```

Applying `softmax` before `CrossEntropyLoss` is the single most common PyTorch beginner bug.
It doesn't crash. It just trains badly, because you've applied softmax twice.

**Predictions** come from the logits directly — the threshold is a decision, separate from
the model:

```python
pred = (logits > 0).long()                # binary, equivalent to p > 0.5
pred = logits.argmax(dim=1)               # multi-class
p = torch.sigmoid(logits)                 # only when you need calibrated probabilities
```

### Multi-class: softmax

$$\text{softmax}(z)_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$$

$C$ outputs instead of 1, probabilities summing to 1, and cross-entropy generalises BCE:
$L = -\log \hat{p}_{y}$, the negative log-probability assigned to the correct class. Note
that only the true class's probability appears — you're not asked to push the others down
explicitly; softmax's normalization does that for you.

## 7. Metrics: accuracy is not enough

For 99% healthy / 1% diseased, "always predict healthy" scores 99%. Know the confusion
matrix:

|  | predicted 0 | predicted 1 |
|---|---|---|
| **actual 0** | TN | FP |
| **actual 1** | FN | TP |

- **Precision** $= TP/(TP+FP)$ — of the things I flagged, how many were right? (cost of false alarms)
- **Recall** $= TP/(TP+FN)$ — of the things that were there, how many did I find? (cost of misses)
- **F1** $= 2PR/(P+R)$ — harmonic mean, when you need one number.
- **IoU / Dice** — the segmentation versions of the same idea (chapter 6).

Precision and recall trade off against each other through the decision threshold, which is
why **ROC-AUC** / average precision (threshold-free summaries) are useful for model
selection. Pick the metric that matches the cost of the mistake, then optimize that.

## 8. Overfitting, and the two curves you always plot

- **Underfitting** — train loss high, val loss high. Model too weak, or LR wrong, or bug.
- **Good fit** — both low, small gap.
- **Overfitting** — train loss keeps falling, val loss turns back up. Memorisation.

Fixes in the order you should reach for them: **more/better data** → **augmentation**
(chapter 4) → **weight decay** → **early stopping** → **smaller model** → **dropout**.

Weight decay (L2) adds $\lambda \|w\|^2$ to the loss, which pulls weights toward zero:
`torch.optim.SGD(..., weight_decay=1e-4)`. It's essentially free, so it's usually on.

Plot train and validation loss every run. The *shape* of those two curves diagnoses more
problems than any single number.

## 9. Why this chapter isn't enough for images

Flatten a 32×32 RGB image and feed it to a linear classifier: 3072 inputs × 10 classes =
30,730 parameters, and it will reach maybe 40% on CIFAR-10. Two fundamental problems:

1. **No spatial structure.** Pixel $(0,0)$ and pixel $(0,1)$ are neighbours; to a linear
   layer they're unrelated coordinates. Shuffle all pixels with a fixed permutation and the
   model performs *exactly the same* — proof that it isn't using geometry at all.
2. **No translation invariance.** A cat shifted three pixels right is, to a flattened linear
   model, a completely different input. It must learn every position independently, from
   scratch.

Convolution fixes both, with *local* connections and *shared* weights. That's chapter 3.

---

## Checklist before moving on

- [ ] Write the 5-step training loop from memory, in order.
- [ ] What happens if you omit `optimizer.zero_grad()`? Why?
- [ ] Why does `nn.CrossEntropyLoss` take logits and not softmax outputs?
- [ ] What dtype and shape do `logits` and `target` need for `CrossEntropyLoss`?
- [ ] Loss went to `nan` on epoch 1. Name the first three things you check.
- [ ] Train loss 0.01, val loss 0.9 and rising — what is happening, and what do you try?
- [ ] Why does a linear model on flattened pixels ignore geometry?

Next: [Chapter 3 — Convolution](03_convolution.md)
