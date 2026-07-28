# Chapter 1 — NumPy & pandas for image data

**Goal:** be fluent enough with arrays that tensor code later reads as obvious. Everything
in deep-learning CV is a shaped block of numbers plus a bookkeeping table of file paths and
labels. NumPy is the first, pandas is the second.

Notebook: [`notebooks/01_numpy_pandas.ipynb`](../notebooks/01_numpy_pandas.ipynb) ·
Exercise: [`exercises/ex01_numpy_pandas.ipynb`](../exercises/ex01_numpy_pandas.ipynb)

---

## 1. An image *is* an array

| Thing | Shape | dtype | Range |
|---|---|---|---|
| Grayscale photo | `(H, W)` | `uint8` | 0–255 |
| RGB photo (as loaded by PIL/matplotlib) | `(H, W, 3)` | `uint8` | 0–255 |
| RGB photo (as PyTorch wants it) | `(3, H, W)` | `float32` | 0.0–1.0 |
| Batch of 32 RGB 224×224 | `(32, 3, 224, 224)` | `float32` | normalized |
| Segmentation mask (5 classes) | `(H, W)` | `int64` | 0–4 |
| Model logits, 10 classes | `(32, 10)` | `float32` | any real |

Two conventions collide constantly:

- **HWC** ("channels last") — PIL, OpenCV, matplotlib, TensorFlow.
- **CHW** ("channels first") — PyTorch, because convolution kernels are laid out that way.

`np.transpose(img, (2, 0, 1))` goes HWC→CHW; `(1, 2, 0)` comes back. `torchvision`'s
`ToTensor()` does HWC-uint8-0..255 → CHW-float32-0..1 in one step, which is why forgetting
it produces images that look like noise.

The batch dimension is always first and is called `N` or `B`. So the full PyTorch image
layout is **NCHW**.

## 2. dtype is not a detail

```python
a = np.array([200, 100], dtype=np.uint8)
a + 100          # array([44, 200], dtype=uint8)   <- 300 wrapped around
a.astype(np.float32) + 100
```

`uint8` overflow silently wraps. Rules of thumb:

- Load as `uint8`, convert to `float32` **before** doing arithmetic.
- Never use `float64` for images — twice the memory, no benefit; GPUs are built for `float32`.
- Class labels are `int64` (`torch.long`), because that's what `nn.CrossEntropyLoss` wants.
- Masks resized with bilinear interpolation get *fractional* class ids. Always use nearest
  neighbour for masks. This bug produces a model that trains but never scores well.

## 3. Views vs copies — the #1 silent bug

Basic slicing returns a **view** into the same memory. Writing to it changes the original.

```python
img = np.zeros((4, 4))
patch = img[1:3, 1:3]   # view
patch[:] = 1            # img is modified too
patch.base is img       # True -> it is a view
```

Fancy (boolean / integer-array) indexing returns a **copy**:

```python
img[img > 0.5]          # copy
img[[0, 2]]             # copy
img[1:3, 1:3].copy()    # explicit copy - do this when you mean it
```

Practical consequence: an augmentation function that slices and writes in place will
quietly corrupt your dataset cache. If a function takes an array and modifies it,
either document that or `copy()` on entry.

`np.shares_memory(a, b)` settles arguments.

## 4. Broadcasting

Align shapes from the **right**; each pair of dims must be equal or one of them 1.

```
image  (256, 256, 3)
mean   (        3,)     -> stretched to (256, 256, 3)   OK
result (256, 256, 3)

image  (3, 256, 256)    # CHW
mean   (        3,)     -> tries to align 3 with 256     ERROR
mean   (3, 1, 1)        -> OK
```

That is the entire reason you see `mean.reshape(-1, 1, 1)` in normalization code. For
per-channel normalization:

```python
img_hwc = img.astype(np.float32) / 255.0
img_hwc = (img_hwc - mean) / std                 # mean.shape == (3,)

img_chw = np.transpose(img_hwc, (2, 0, 1))
img_chw = (img_chw - mean[:, None, None]) / std[:, None, None]
```

`None` (a.k.a. `np.newaxis`) inserts a length-1 axis. `img[None]` adds a batch dim —
the NumPy equivalent of `tensor.unsqueeze(0)`.

## 5. Axes and reductions

`axis` = the axis that **disappears**.

```python
x = np.random.rand(32, 3, 64, 64)      # NCHW batch

x.mean()                  # ()          scalar over everything
x.mean(axis=(0, 2, 3))    # (3,)        per-channel mean over batch+space  <- dataset stats
x.mean(axis=1)            # (32,64,64)  grayscale-ish, channels collapsed
x.mean(axis=(2, 3))       # (32,3)      global average pooling
x.sum(axis=1, keepdims=True)  # (32,1,64,64)  keepdims keeps it broadcastable
```

`keepdims=True` is what lets you write `x / x.sum(axis=1, keepdims=True)` (that's a softmax
denominator). Without it the shapes don't line up.

For predictions: `logits.argmax(axis=1)` turns `(N, C)` logits into `(N,)` predicted
classes; for segmentation, `(N, C, H, W)` → `(N, H, W)` with the same call. Same idea,
different rank — this is the payoff of thinking in axes rather than in loops.

## 6. Shape surgery

| Operation | Use |
|---|---|
| `reshape(-1)` / `ravel()` | flatten an image into a feature vector |
| `reshape(N, -1)` | flatten a batch, keep the batch dim — exactly what `nn.Flatten()` does |
| `transpose(2,0,1)` | HWC→CHW (returns a view, may be non-contiguous) |
| `np.stack(list, axis=0)` | glue individual images into a batch (adds an axis) |
| `np.concatenate(list, axis=0)` | glue batches together (no new axis) |
| `img[None]` / `squeeze()` | add / drop length-1 axes |
| `np.pad(img, ((1,1),(1,1)), mode='reflect')` | padding before convolution |

`reshape` never reorders data, it reinterprets strides. `transpose` reorders the *view*
of the axes. Confusing the two gives you scrambled images: `img.reshape(3, H, W)` on an
HWC array is **not** CHW, it's garbage. Use `transpose`.

## 7. Masks, thresholds, and counting

```python
mask = gray > 0.5                       # bool array, same shape
mask.sum()                              # pixel count (True == 1)
mask.mean()                             # fraction of the image
np.where(mask, 255, 0).astype(np.uint8) # bool -> displayable
img[mask] = 0                           # blank the selected pixels

ids, counts = np.unique(mask_multiclass, return_counts=True)   # class balance
```

`np.unique(..., return_counts=True)` on a mask is how you discover that 94% of your pixels
are background — which is why plain accuracy is a useless segmentation metric (chapter 6).

`np.clip(x, 0, 1)` after any arithmetic on images. `np.isnan(x).any()` when a loss goes NaN.

## 8. Vectorize, don't loop

```python
# slow: ~seconds for a 512x512 image
for i in range(H):
    for j in range(W):
        out[i, j] = img[i, j] * 1.2

out = img * 1.2      # fast: one C loop under the hood
```

A Python-level loop over pixels is ~100–1000× slower. In chapter 3 you *will* write a
convolution with explicit loops — once, to understand it — and then immediately replace it
with a vectorized version to feel the difference.

`np.einsum` is worth knowing for the cases broadcasting can't express:

```python
np.einsum('nchw,nchw->n', a, b)    # per-sample dot product
np.einsum('ij,jk->ik', A, B)       # matrix multiply
```

## 9. Random numbers, reproducibly

Use the modern `Generator` API, not the legacy global `np.random.seed`:

```python
rng = np.random.default_rng(seed=0)
rng.random((4, 4))
rng.integers(0, 256, size=(8, 8), dtype=np.uint8)
rng.normal(0, 1, size=100)
rng.permutation(10)                 # shuffling indices for a split
```

Passing an explicit `rng` around makes a function testable. A seeded run that you can
re-run identically is the difference between "the change helped" and "we got lucky".

## 10. pandas: the bookkeeping half

A CV dataset in practice is a folder of images plus a table:

```
path                        label   split   width  height
data/train/cat/0001.jpg     cat     train   500    375
data/train/dog/0002.jpg     dog     train   640    480
```

That table is a `DataFrame`, and it drives your `Dataset.__getitem__`. The operations
you'll use over and over:

```python
df = pd.read_csv('manifest.csv')

df.head(); df.shape; df.dtypes; df.info(); df.describe()

df['label']                       # Series (one column)
df[['path', 'label']]             # DataFrame (list of columns)
df.loc[3, 'label']                # by label/index
df.iloc[3, 1]                     # by integer position
df[df.label == 'cat']             # boolean mask
df.query('width > 400 and label == "cat"')

df['label'].value_counts()        # class balance - always check this
df['area'] = df.width * df.height # vectorized new column
df.groupby('label')['area'].agg(['count', 'mean', 'std'])
df.merge(other, on='path', how='left')
df.isna().sum()                   # missing values per column
df.sort_values('area', ascending=False).head()
df['label'].map({'cat': 0, 'dog': 1})   # label -> class id
```

Two habits worth forming:

1. **`value_counts()` before you train.** Imbalance you didn't know about explains most
   "why is my accuracy 50%" mysteries.
2. **Stratified splits.** A random split can hand you a validation set with no examples of
   a rare class. Group by label, shuffle within group, take a fixed fraction from each:

```python
def stratified_split(df, label_col='label', val_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    df = df.copy()
    df['split'] = 'train'
    for _, idx in df.groupby(label_col).groups.items():
        idx = np.array(idx)
        rng.shuffle(idx)
        n_val = int(round(val_frac * len(idx)))
        df.loc[idx[:n_val], 'split'] = 'val'
    return df
```

Avoid `df.apply(..., axis=1)` on large frames — it's a Python loop in disguise. Use
vectorized column arithmetic, `map`, or `np.where` instead. `apply` is fine on a few
thousand rows of metadata; it is not fine per pixel.

### `SettingWithCopyWarning`

Same view-vs-copy problem as NumPy, different message. `df[df.a > 1]['b'] = 0` modifies a
temporary. Write `df.loc[df.a > 1, 'b'] = 0` instead.

---

## Checklist before moving on

You should be able to answer these without running code:

- [ ] Shape of a batch of 16 RGB 128×128 images in PyTorch layout? *(16, 3, 128, 128)*
- [ ] `x.shape == (8, 3, 32, 32)`; what is `x.mean(axis=(0, 2, 3)).shape`? *(3,)*
- [ ] Why does `(img_chw - mean)` fail when `mean.shape == (3,)`?
- [ ] Which of `img[0:2]`, `img[img > 0]`, `img[[0, 1]]` return views?
- [ ] How do you turn `(N, 10)` logits into predicted labels?
- [ ] Why must masks be resized with nearest-neighbour interpolation?
- [ ] Why is `df.loc[mask, 'col'] = v` correct and `df[mask]['col'] = v` not?

Then do the exercise. Next: [Chapter 2 — ML foundations](02_ml_foundations.md)
