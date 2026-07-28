# Chapter 3 — Convolution: the operation that makes CV work

**Goal:** understand convolution well enough to design a CNN on paper — predict every output
shape, count every parameter, and know what a layer is looking for.

Notebook: [`notebooks/03_convolution_image_ops.ipynb`](../notebooks/03_convolution_image_ops.ipynb) ·
Exercise: [`exercises/ex03_convolution.ipynb`](../exercises/ex03_convolution.ipynb)

---

## 1. What convolution is

Slide a small window (the **kernel**, or filter) over the image. At each position, multiply
elementwise and sum. That single number is one output pixel.

$$\text{out}[i, j] = \sum_{m}\sum_{n} \text{in}[i + m,\; j + n] \cdot K[m, n]$$

Three properties fall out of this, and they are exactly what chapter 2's linear model lacked:

1. **Locality** — an output pixel depends only on a small neighbourhood. Nearby pixels are
   related; distant ones are composed later, through depth.
2. **Weight sharing** — the *same* kernel is applied everywhere. A 3×3 kernel has 9 weights
   whether the image is 32×32 or 4000×3000.
3. **Translation equivariance** — shift the input, and the output shifts identically. An edge
   detector learned in the top-left corner works in the bottom-right for free.

That third point is why a CNN needs orders of magnitude less data than a fully connected net
for the same visual task: it doesn't have to relearn "cat" at every position.

### The parameter count argument

Connect a 32×32×3 image to 32×32×16 outputs:

| Layer | Parameters |
|---|---|
| Fully connected | $3072 \times 16384 \approx 50{,}000{,}000$ |
| Conv 3×3, 3→16 channels | $3 \times 3 \times 3 \times 16 + 16 = 448$ |

Five orders of magnitude, and the conv version generalises *better*. This is the single most
important trade in computer vision.

### Cross-correlation, actually

What every deep learning framework calls "convolution" is technically **cross-correlation** —
true mathematical convolution flips the kernel first. It makes no practical difference: the
kernel is learned, so it learns whichever orientation it needs. But it matters when you
compare against `scipy.signal.convolve2d`, which *does* flip. Use
`scipy.signal.correlate2d` to match PyTorch.

## 2. The output size formula

Memorise this. You will use it constantly:

$$H_{out} = \left\lfloor \frac{H_{in} + 2p - d(k-1) - 1}{s} \right\rfloor + 1$$

with kernel $k$, padding $p$, stride $s$, dilation $d$. For the common case $d=1$:

$$H_{out} = \left\lfloor \frac{H_{in} + 2p - k}{s} \right\rfloor + 1$$

Consequences worth knowing by heart:

| Setting | Effect |
|---|---|
| $k=3, s=1, p=1$ | **size preserved**. The default building block |
| $k=5, s=1, p=2$ | size preserved ($p = (k-1)/2$ for odd $k$) |
| $k=3, s=2, p=1$ | **halves** the size — downsampling by conv |
| $k=1, s=1, p=0$ | size preserved, mixes channels only |
| $k=2, s=2, p=0$ | halves the size (this is max-pool's geometry) |

**Odd kernels are conventional** because they have a well-defined centre, so
`p = (k-1)//2` keeps the size exactly and doesn't shift the image by half a pixel.

Padding modes: `zeros` (default), `reflect`, `replicate`, `circular`. Zero padding darkens
borders slightly, which the network learns to ignore; `reflect` avoids that and is a common
choice in image restoration.

## 3. Channels: the part people get wrong

A conv layer's kernel is **not** 2D. For `Conv2d(C_in, C_out, k)` the weight tensor is:

$$W \in \mathbb{R}^{C_{out} \times C_{in} \times k \times k}$$

Each of the $C_{out}$ filters spans **all** input channels, and produces **one** output
channel by summing across them:

$$\text{out}[c_{out}, i, j] = b_{c_{out}} + \sum_{c_{in}} \sum_{m,n} \text{in}[c_{in}, i+m, j+n] \cdot W[c_{out}, c_{in}, m, n]$$

So the parameter count is $C_{out} \times C_{in} \times k \times k + C_{out}$ (the `+C_out`
is the bias, one per output channel).

Quick check — `Conv2d(3, 64, kernel_size=7)`: $64 \times 3 \times 7 \times 7 + 64 = 9472$.
That's ResNet's first layer.

Two special cases worth naming:

- **1×1 convolution.** No spatial context at all; it's a per-pixel linear layer across
  channels. Used to change channel count cheaply (bottlenecks) and to mix features.
- **Depthwise convolution** (`groups=C_in`). Each input channel gets its own kernel, no
  cross-channel mixing. Combined with a 1×1 conv ("depthwise separable") it approximates a
  full conv at ~1/8 the cost. This is what makes MobileNet small.

## 4. Classic kernels, and what learning replaces

Before CNNs, you designed these by hand. Now they're what the first layer learns anyway —
which is a nice sanity check on the whole idea.

| Kernel | Effect |
|---|---|
| $\frac{1}{9}\begin{bmatrix}1&1&1\\1&1&1\\1&1&1\end{bmatrix}$ | box blur (mean) |
| $\frac{1}{16}\begin{bmatrix}1&2&1\\2&4&2\\1&2&1\end{bmatrix}$ | Gaussian blur — smoother, no ringing |
| $\begin{bmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{bmatrix}$ | Sobel x — **vertical** edges |
| $\begin{bmatrix}-1&-2&-1\\0&0&0\\1&2&1\end{bmatrix}$ | Sobel y — **horizontal** edges |
| $\begin{bmatrix}0&-1&0\\-1&4&-1\\0&-1&0\end{bmatrix}$ | Laplacian — all edges, second derivative |
| $\begin{bmatrix}0&-1&0\\-1&5&-1\\0&-1&0\end{bmatrix}$ | sharpen (identity + Laplacian) |

Two useful facts:

- **Sums matter.** A kernel summing to 1 preserves brightness (blurs). A kernel summing to 0
  responds to *change* and gives ~0 on flat regions (edge detectors). That's why edge maps
  are mostly black.
- **Gradient magnitude** $\sqrt{G_x^2 + G_y^2}$ from the two Sobels is orientation-independent
  edge strength — the basis of Canny, HOG, and SIFT. A CNN's first layer reliably learns
  something very close to a bank of oriented Sobel filters. You'll see it in chapter 5.

## 5. Pooling and downsampling

**Max pool 2×2, stride 2** takes the largest value in each 2×2 block: quarter the spatial
size, same channel count, no parameters.

Why downsample at all?

- **Receptive field growth.** After halving, a 3×3 kernel covers twice as much of the
  original image.
- **Compute.** Halving H and W quarters the work in every later layer.
- **Small-shift tolerance.** Max pool's output is unchanged if the max moves within its window.

Max vs average: max keeps the strongest response ("was this feature present anywhere here?"),
average keeps the mean ("how much of it overall?"). Max is the default in classifiers;
**global average pooling** (`AdaptiveAvgPool2d(1)`, reducing H×W to 1×1) is the standard
modern way to end a CNN before the classifier head, because it's shape-agnostic and has no
parameters.

Modern architectures increasingly drop pooling for **strided convolution** (`stride=2`),
which learns *how* to downsample instead of hard-coding it. Both are common; know both.

## 6. Receptive field

The receptive field is the region of the *input* that influences one output value. Stacking
grows it:

| Layers of 3×3, stride 1 | Receptive field |
|---|---|
| 1 | 3×3 |
| 2 | 5×5 |
| 3 | 7×7 |
| 5 | 11×11 |

Linear growth: $\text{RF} = 1 + \sum_l (k_l - 1)\prod_{i<l} s_i$. With stride and pooling the
product term makes it grow *multiplicatively*, which is how a ~20-layer net gets a receptive
field covering the whole image.

**Two 3×3 convs beat one 5×5.** Same 5×5 receptive field, but $2 \times 9 = 18$ weights per
channel pair instead of 25, and a nonlinearity in between. That observation is the entire
design of VGG, and it's why you rarely see kernels bigger than 3×3 in the middle of a network.

**Practical consequence:** if your object is 40 pixels wide and your network's receptive field
is 15 pixels, no amount of training will let it see the whole object at once. Downsample more,
add depth, or use dilation.

**Dilated (atrous) convolution** inserts gaps in the kernel: a 3×3 with `dilation=2` covers
5×5 using 9 weights. It grows the receptive field *without* losing resolution, which is why
segmentation networks (DeepLab) use it heavily — see chapter 6.

## 7. From convolution to a CNN block

The canonical unit, repeated everywhere since ~2015:

```python
nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False)
nn.BatchNorm2d(c_out)
nn.ReLU(inplace=True)
```

- **Order matters** and this is the standard one (conv → norm → activation).
- **`bias=False` before BatchNorm.** BN subtracts a learned mean, so the conv's bias is
  mathematically redundant — it would just be cancelled. Harmless but wasteful.
- **Why BatchNorm.** It normalizes each channel over the batch to zero mean and unit variance,
  then rescales with learned parameters. Effect: much less sensitivity to initialization and
  learning rate, faster convergence, and a mild regularizing effect from batch noise. Its
  train/eval behaviour differs (batch statistics vs running averages) — this is the concrete
  reason `model.eval()` matters.
- **Why ReLU** (`max(0, x)`): cheap, and no vanishing-gradient problem for positive inputs.
  Without a nonlinearity between convs, two stacked convs collapse into a single linear
  operation and depth buys you nothing.

A whole classifier is then just: a few of these blocks with downsampling between them,
global average pooling, and a linear layer.

```
[Conv-BN-ReLU] x2 -> pool -> [Conv-BN-ReLU] x2 -> pool -> ... -> GAP -> Linear(n_classes)
      32x32                        16x16                            1x1
```

Channels typically double when spatial size halves, keeping the compute per stage roughly
constant. That's the pattern in chapter 4.

## 8. Implementation: how it's really computed

Nobody loops. Convolution is turned into **one matrix multiply** (`im2col` / "unfold"):

1. Extract every $k \times k \times C_{in}$ patch and flatten it into a row → matrix
   $(H_{out}W_{out}) \times (C_{in}k^2)$.
2. Flatten the kernels into $(C_{in}k^2) \times C_{out}$.
3. Multiply. Reshape the result to $(C_{out}, H_{out}, W_{out})$.

This trades memory for speed, and it's why GPUs are so good at CNNs: everything reduces to
GEMM, the single most optimized routine in computing. `torch.nn.functional.unfold` exposes
step 1 if you want to see it (the notebook does).

You will never write this in production. Implementing it once is how you stop treating
`Conv2d` as a black box.

---

## Checklist before moving on

- [ ] `Conv2d(3, 32, kernel_size=3, padding=1)` on a `(8, 3, 64, 64)` input — output shape? parameters?
- [ ] Which `(k, s, p)` preserves spatial size? Which halves it?
- [ ] Why is the weight tensor 4D and not 2D?
- [ ] Why does an edge-detection kernel sum to zero?
- [ ] Receptive field after three 3×3 stride-1 convs?
- [ ] Why two 3×3 convs instead of one 5×5?
- [ ] Why `bias=False` when a BatchNorm follows?
- [ ] What does a 1×1 convolution actually do?

*(Answers: `(8, 32, 64, 64)`, 896 params; `k=3,s=1,p=1` preserves, `k=3,s=2,p=1` halves;
because each filter spans all input channels; so it responds to change, not brightness;
7×7; fewer parameters plus an extra nonlinearity; because BN's learned shift makes it
redundant; a per-pixel linear map across channels.)*

Next: [Chapter 4 — CNN classification](04_cnn_classification.md)
