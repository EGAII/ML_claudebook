# Chapter 7 — GANs: learning to generate

**Goal:** understand generative modelling, and train a GAN that produces recognisable images.
This is your first model whose output is an *image* rather than a label.

> **GPU runtime required** (Runtime → Change runtime type → T4 GPU).

Notebook: [`notebooks/07_gan_dcgan.ipynb`](../notebooks/07_gan_dcgan.ipynb) ·
Exercise: [`exercises/ex07_gan.ipynb`](../exercises/ex07_gan.ipynb)

---

## 1. Discriminative vs generative

Everything in chapters 2–6 was **discriminative**: learn $p(y \mid x)$ — given an image, produce
a label. Generative modelling turns it around: learn $p(x)$ itself, well enough to **sample new
$x$** that look like they came from the data.

That's a much harder problem, and it changes three things fundamentally:

| | Discriminative (ch 2–6) | Generative (ch 7–8) |
|---|---|---|
| Target | a label you were given | no target — there's no "correct" output |
| Loss | compare prediction to truth | compare *distributions* |
| Evaluation | accuracy, IoU | no ground truth to compare against |

That last row is the real difficulty. When a classifier is wrong you can point at the label. When a
generator produces a slightly-off face, there is nothing to subtract it from. **Defining the
objective is the whole problem** — and GANs answer it in a genuinely clever way.

## 2. The GAN idea

Two networks, trained against each other:

- **Generator** $G(z)$ takes random noise $z \sim \mathcal{N}(0, I)$ and outputs an image.
- **Discriminator** $D(x)$ takes an image and outputs the probability it is real.

$G$ tries to fool $D$; $D$ tries not to be fooled. Formally a minimax game:

$$\min_G \max_D \; \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z}[\log(1 - D(G(z)))]$$

The insight: **we can't write down a loss for "looks like a real image", so we learn one.** $D$ *is*
the loss function, and it improves as $G$ improves. That's why GANs produced sharp images years
before anything else did — a fixed pixel-wise loss like MSE would just produce blurry averages.

$z$ lives in a **latent space** (typically 100 dimensions). Training makes $G$ a map from that
simple Gaussian to the complicated manifold of real images. Because the map is smooth, walking a
straight line in $z$ gives you a smooth *semantic* morph between two generated images — one of the
most convincing demonstrations that the model learned structure rather than memorised pixels.

### Implementing the objective

You never write the minimax formula. Both halves are just binary cross-entropy with different
labels:

```python
bce = nn.BCEWithLogitsLoss()

# --- discriminator: real -> 1, fake -> 0
d_real = bce(D(real), torch.ones(n, 1))
d_fake = bce(D(fake.detach()), torch.zeros(n, 1))     # detach: don't backprop into G here
d_loss = d_real + d_fake

# --- generator: make D say 1 for fakes
g_loss = bce(D(fake), torch.ones(n, 1))               # NOT -bce(..., zeros)
```

Two details that matter enormously:

**`.detach()` on the fake in the D step.** Without it, the D loss backpropagates into $G$'s weights.
Since you then call `g_optimizer.step()`, $G$ gets updated by the discriminator's objective — which
is the opposite of what you want. It doesn't crash. Training just never works.

**The non-saturating generator loss.** The original minimax says $G$ minimizes
$\log(1 - D(G(z)))$. Early in training $D$ easily rejects fakes, $D(G(z)) \approx 0$, and that
term's gradient vanishes — exactly when $G$ most needs signal. So in practice $G$ *maximizes*
$\log D(G(z))$ instead, which is `bce(D(fake), ones)`. Same fixed point, non-vanishing gradient.
Every implementation does this; it's rarely explained.

## 3. DCGAN: the architecture that made it work

The 2016 DCGAN paper's guidelines still hold for small images:

**Generator** — noise vector to image, upsampling as you go:

```python
nn.ConvTranspose2d(z_dim, 256, 4, 1, 0)   # (N,z,1,1) -> (N,256,4,4)
nn.BatchNorm2d(256); nn.ReLU()
nn.ConvTranspose2d(256, 128, 4, 2, 1)     # -> 8x8
nn.BatchNorm2d(128); nn.ReLU()
nn.ConvTranspose2d(128, 64, 4, 2, 1)      # -> 16x16
nn.BatchNorm2d(64); nn.ReLU()
nn.ConvTranspose2d(64, 1, 4, 2, 1)        # -> 32x32
nn.Tanh()                                  # output in [-1, 1]
```

**Discriminator** — a normal CNN classifier with one output logit:

```python
nn.Conv2d(1, 64, 4, 2, 1); nn.LeakyReLU(0.2)               # 32 -> 16, no BN on the first layer
nn.Conv2d(64, 128, 4, 2, 1); nn.BatchNorm2d(128); nn.LeakyReLU(0.2)
nn.Conv2d(128, 256, 4, 2, 1); nn.BatchNorm2d(256); nn.LeakyReLU(0.2)
nn.Conv2d(256, 1, 4, 1, 0)                                  # -> (N,1,1,1) logit, NO sigmoid
```

The rules, and why:

| Rule | Reason |
|---|---|
| `Tanh` output, data normalized to $[-1,1]$ | symmetric range, saturating output; **the generator's range must match the data's** |
| `LeakyReLU(0.2)` in $D$ | plain ReLU kills gradient on negative inputs, and $D$'s gradient is $G$'s only teacher |
| No BatchNorm on $D$'s first layer | it would normalize away the real/fake statistics $D$ needs |
| `4×4` kernel, stride 2 | divisible by the stride, so no checkerboard artifacts |
| No pooling | learned strided (transposed) convolution instead |
| No sigmoid on $D$ | use `BCEWithLogitsLoss` (chapter 2's rule, still) |
| `Adam(lr=2e-4, betas=(0.5, 0.999))` | the standard GAN setting; $\beta_1 = 0.5$ damps momentum, which destabilises adversarial training |

## 4. Failure modes — this is the chapter's real content

GANs fail in ways nothing else in this course does. Recognising the failure from the symptom is the
skill.

| Symptom | Name / cause | Fix |
|---|---|---|
| All samples look identical | **mode collapse** — $G$ found one image that fools $D$ | lower $G$'s LR, add noise, minibatch discrimination, use WGAN-GP |
| $D$ loss → 0, $G$ loss → ∞ | $D$ won; $G$ gets no usable gradient | weaken $D$ (fewer params, lower LR, label smoothing) |
| Both losses flat, samples are noise | $D$ too weak to teach anything | strengthen $D$, train it more steps per $G$ step |
| Samples oscillate between modes | non-convergence — the game cycles | lower LRs, $\beta_1=0.5$, EMA of $G$'s weights |
| `nan` after a while | exploding gradients | gradient clipping, lower LR, spectral norm |

**The one thing to internalise: GAN loss curves tell you almost nothing about sample quality.** A
perfectly balanced GAN sits at $D_{\text{loss}} \approx 1.386 = 2\ln 2$ and $D$ accuracy ≈ 0.5
forever, because the two objectives are *relative* to each other. Loss going down is not progress.
**You evaluate a GAN by looking at samples** — and by FID.

Useful stabilisers, roughly in order of value:

1. **One-sided label smoothing** — train $D$ with real labels at 0.9 instead of 1.0. Stops $D$
   becoming overconfident, which is what starves $G$.
2. **Track $D$'s accuracy**, not its loss. It should hover near 0.5–0.7. Pinned at 1.0 means $D$ won.
3. **Spectral normalization** on $D$ (`nn.utils.parametrizations.spectral_norm`) — constrains its
   Lipschitz constant. Nearly free, very effective.
4. **WGAN-GP** — replaces BCE with a Wasserstein distance plus a gradient penalty. More stable, and
   its loss actually correlates with quality. The main alternative formulation worth knowing.
5. **EMA of the generator weights** for sampling. Standard in modern generative work.

## 5. Evaluating a generative model

Two things matter and they trade off: **fidelity** (do samples look real?) and **diversity** (do
they cover the data?). Mode collapse is perfect fidelity with zero diversity, which is why any
single-number metric must penalise it.

**FID** (Fréchet Inception Distance) is the standard. Push real and generated images through a
pretrained Inception network, take the 2048-d features, fit a Gaussian to each, and measure the
distance between them:

$$\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$$

Lower is better. It captures both failure modes: wrong-looking samples move $\mu$, collapsed samples
shrink $\Sigma$. Caveats worth knowing:

- It needs **thousands** of samples (≥10k for a stable number); computing it on 100 is meaningless.
- The absolute value is only comparable within an identical setup — same sample count, same
  Inception weights, same preprocessing. Never compare your FID to a paper's.
- On non-natural images (grayscale digits, medical scans) the ImageNet features are a poor basis, so
  treat it as a relative signal only.

Also worth knowing: **Inception Score** (older, doesn't compare to real data, largely superseded),
and **precision/recall for generative models** (which separates fidelity from diversity instead of
mixing them into one number).

## 6. Conditional GANs

Unconditional $G(z)$ gives you *a* digit. Usually you want to choose *which*. Feed the class label
to both networks:

```python
# generator: concatenate a label embedding to the noise
z_and_y = torch.cat([z, self.label_emb(y)], dim=1)

# discriminator: give it the label too, as extra channels
y_map = self.label_emb(y).view(n, -1, 1, 1).expand(-1, -1, H, W)
d_in = torch.cat([x, y_map], dim=1)
```

$D$ must see the label as well, or nothing forces $G$ to respect it — $D$ would happily accept a
realistic 3 labelled "7". This "condition both players" pattern is the basis of **pix2pix** (condition
on an input image), **CycleGAN** (unpaired image translation), and text-to-image models.

## 7. Where GANs stand now

| Model | Contribution |
|---|---|
| DCGAN (2016) | the conv architecture that made GAN training feasible |
| WGAN / WGAN-GP (2017) | a better-behaved distance; meaningful loss |
| pix2pix / CycleGAN (2017) | paired and unpaired image-to-image translation |
| Progressive GAN / StyleGAN (2018–20) | high-resolution photorealistic faces; a genuinely interpretable latent space |
| BigGAN (2019) | large-scale class-conditional ImageNet |

Diffusion models (chapter 8) have largely displaced GANs for image *generation* — they're far easier
to train and cover the data distribution better. But GANs are still competitive where **inference
speed** matters: a GAN generates in one forward pass, diffusion needs tens to thousands. That's why
GANs persist in real-time super-resolution, audio vocoders, and on-device generation. And the
adversarial *loss* is still widely used as one term among several (e.g. in the decoders of latent
diffusion models — chapter 8).

---

## Checklist before moving on

- [ ] Why can't you train a generator with plain MSE against real images?
- [ ] Write both loss terms with `BCEWithLogitsLoss`. What labels does each use?
- [ ] Why `.detach()` the fake batch in the discriminator step?
- [ ] What is the non-saturating generator loss, and what problem does it solve?
- [ ] Why must the generator's output activation match the data's normalization range?
- [ ] Your samples are all nearly identical. What is this called, and what do you try?
- [ ] $D$'s loss is 0.001 and $G$'s is 12. Who is winning, and what do you do?
- [ ] Why is a falling GAN loss not evidence of progress?
- [ ] Why must the discriminator also receive the label in a conditional GAN?

*(Answers: MSE against *which* real image? — averaging many valid outputs gives blur, and there is no
per-sample target; D uses real→1, fake→0, G uses fake→1; because the D loss would otherwise update G
with D's objective; G maximizes log D(G(z)) rather than minimizing log(1−D(G(z))), which vanishes when
D is winning; a tanh generator can never produce values outside [−1,1], so data in [0,1] is
unreachable and vice versa; mode collapse — lower G's LR, label smoothing, spectral norm, WGAN-GP; D
has won and G's gradient has vanished — weaken D or smooth its labels; because the two losses are
defined relative to an opponent that is itself changing; otherwise nothing penalises G for ignoring
the label.)*

Next: [Chapter 8 — Diffusion models](08_diffusion.md)
