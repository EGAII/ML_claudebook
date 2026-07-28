# Chapter 8 — Diffusion models: generation as denoising

**Goal:** implement DDPM from scratch, understand why its loss is a plain MSE, and see the U-Net
from chapter 6 become a generative model.

> **GPU runtime required** (Runtime → Change runtime type → T4 GPU).

Notebook: [`notebooks/08_diffusion_ddpm.ipynb`](../notebooks/08_diffusion_ddpm.ipynb) ·
Exercise: [`exercises/ex08_diffusion.ipynb`](../exercises/ex08_diffusion.ipynb)

---

## 1. The idea

Destroying an image is easy: add a little Gaussian noise, repeatedly, until nothing is left. That
process has no parameters and no failure modes.

Diffusion models learn to **run it backwards**. If a network can remove a small amount of noise from
a slightly-noisy image, then starting from pure noise and applying it many times produces an image.

That reframing is the whole trick, and it buys something remarkable: **generation becomes supervised
learning.** We can *construct* training pairs — take a real image, add a known amount of noise, ask
the network to predict the noise we added. There's a ground-truth target, so the loss is a plain MSE
and none of chapter 7's adversarial instability exists.

| | GAN (ch 7) | Diffusion (ch 8) |
|---|---|---|
| Loss | learned, adversarial | **MSE against known noise** |
| Training | two networks, unstable | one network, boringly stable |
| Loss curve | uninformative | **actually goes down** |
| Mode coverage | drops modes | covers the distribution |
| Sampling | 1 forward pass | 20–1000 forward passes |

That last row is the tradeoff, and it's the only one GANs win.

## 2. The forward process

Add Gaussian noise over $T$ steps according to a **variance schedule** $\beta_1 \dots \beta_T$
(small, increasing, e.g. $10^{-4}$ to $0.02$):

$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\ \sqrt{1-\beta_t}\, x_{t-1},\ \beta_t I\right)$$

The $\sqrt{1-\beta_t}$ shrinks the signal as the noise grows, which keeps the variance bounded — after
many steps you land on $\mathcal{N}(0, I)$ rather than something that blows up.

**The key result** — you never need to iterate. With $\alpha_t = 1 - \beta_t$ and
$\bar\alpha_t = \prod_{s \le t} \alpha_s$, composing Gaussians gives a closed form:

$$\boxed{\;x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I)\;}$$

One line, any $t$, no loop. This is what makes training cheap: sample a random $t$ per image and jump
straight there.

Read it as an interpolation: at $t=0$, $\bar\alpha \approx 1$ and you have the image; at $t=T$,
$\bar\alpha \approx 0$ and you have pure noise. $\bar\alpha_t$ *is* the signal-to-noise ratio, and
the schedule is your choice of how fast to destroy information.

## 3. Training: predict the noise

The network takes a noisy image and the timestep, and predicts the noise that was added:

$$\mathcal{L} = \mathbb{E}_{x_0,\ \varepsilon,\ t}\left[\big\| \varepsilon - \varepsilon_\theta(x_t, t)\big\|^2\right]$$

That's it. The full algorithm:

```python
for x0 in data:
    t = torch.randint(0, T, (batch,))                  # random timestep PER IMAGE
    eps = torch.randn_like(x0)
    xt = sqrt_ab[t] * x0 + sqrt_1mab[t] * eps          # the closed form above
    loss = F.mse_loss(model(xt, t), eps)               # predict the noise
    loss.backward(); optimizer.step()
```

Six lines, and the loss is `mse_loss`. Compare with chapter 7.

**Why predict $\varepsilon$ rather than $x_0$?** Both are valid parameterizations and algebraically
interchangeable — given $x_t$ and one of them you can compute the other. Predicting the noise works
much better in practice: it keeps the target's scale constant across all $t$ (it's always unit
Gaussian), whereas $x_0$-prediction is trivial at small $t$ and near-impossible at large $t$, so the
loss is dominated by whichever end you didn't want. It also happens to make the objective a weighted
variational bound on the likelihood, which is where the original derivation comes from.

**Random $t$ per image, not per batch.** Per-batch works but gives you far noisier gradients — each
step then teaches the network about only one noise level.

## 4. Sampling: run it backwards

Start from $x_T \sim \mathcal{N}(0, I)$ and step down. Each step subtracts the predicted noise
(scaled) and adds a little fresh noise back:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\, \varepsilon_\theta(x_t, t)\right) + \sigma_t z, \qquad z \sim \mathcal{N}(0, I)$$

with $\sigma_t = \sqrt{\beta_t}$ and **no noise added on the final step** ($t = 0$).

```python
x = torch.randn(n, 1, 32, 32, device=device)
for t in reversed(range(T)):
    eps = model(x, torch.full((n,), t, device=device))
    x = (x - beta[t] / sqrt_1mab[t] * eps) / sqrt_alpha[t]
    if t > 0:
        x = x + beta[t].sqrt() * torch.randn_like(x)
```

Why add noise back at all? Without it you'd be doing deterministic gradient-descent-like steps toward
the mean of the data — every sample would converge to the same blurry average, which is exactly
chapter 7's MSE failure. The injected noise is what makes the sampler explore the distribution
instead of collapsing to its mean.

**This loop is why diffusion is slow.** $T = 1000$ means 1000 forward passes per image.

### DDIM: fewer steps

DDIM reinterprets the reverse process as a deterministic ODE, which lets you **skip timesteps**:

$$x_{t-1} = \sqrt{\bar\alpha_{t-1}} \underbrace{\left(\frac{x_t - \sqrt{1-\bar\alpha_t}\,\varepsilon_\theta}{\sqrt{\bar\alpha_t}}\right)}_{\text{predicted } x_0} + \sqrt{1-\bar\alpha_{t-1}}\, \varepsilon_\theta$$

Same trained model, no retraining. 50 steps instead of 1000 — a 20× speedup for a small quality
cost. It's also **deterministic**, so a given noise seed always gives the same image, which is what
makes latent-space interpolation possible (as in chapter 7) and what "seed" means in a
text-to-image UI.

Modern samplers (DPM-Solver, Euler-a, etc.) push this to 10–20 steps. They are all better ODE
solvers for the same learned function.

## 5. Noise schedules

$\beta_t$ is a design choice with real consequences:

- **Linear** (the original DDPM): $\beta$ from $10^{-4}$ to $0.02$ over 1000 steps. Works, but
  destroys information too fast at the end — the last ~20% of steps are nearly pure noise and
  contribute little.
- **Cosine** (improved DDPM): $\bar\alpha_t = \cos^2\!\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)$.
  Keeps signal alive longer, spends more steps in the informative middle. Noticeably better,
  especially at low resolution.

Plot $\bar\alpha_t$ for both — the difference is obvious and it explains the quality gap. A good
diagnostic: look at $x_t$ at $t = 0, T/4, T/2, 3T/4, T$ and check that the middle is *genuinely
ambiguous*. If your image is unrecognisable by $t = T/4$, your schedule is wasting three quarters of
its capacity.

## 6. The architecture

Any image-to-image network works; in practice it's always a **U-Net** (chapter 6), because you need
full-resolution output plus global context. Two additions:

**Time conditioning.** The network must know *how much* noise to remove. Encode $t$ with sinusoidal
embeddings (the transformer positional encoding), pass through a small MLP, and add it to each
residual block's features:

```python
def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

# inside a residual block, after the first conv:
h = h + self.time_mlp(t_emb)[:, :, None, None]     # broadcast over H, W
```

Why not just pass `t` as a number? A single scalar into a conv is a very weak signal, and the network
needs to distinguish 1000 distinct levels precisely. Sinusoidal features at many frequencies give it
a rich, smooth code — the same reason transformers use them for position.

**Self-attention at low resolution** (typically 16×16 and below) for global coherence. Optional at
32×32, essential at 256×256.

Also standard: residual blocks rather than plain convs, GroupNorm rather than BatchNorm (batch-size
independent, and diffusion often trains at small batch sizes), and SiLU activations.

## 7. Conditioning and classifier-free guidance

To generate a *chosen* class, add the class embedding to the time embedding — that's all:

```python
emb = self.time_mlp(t_emb) + self.class_emb(y)
```

Then there's a trick that transformed generative modelling. **Classifier-free guidance**: during
training, randomly drop the label (say 10% of the time) and replace it with a "null" token. The one
network thus learns both the conditional and unconditional prediction. At sampling time, extrapolate
away from the unconditional:

$$\tilde\varepsilon = \varepsilon_\theta(x_t, t, \varnothing) + w \cdot \big(\varepsilon_\theta(x_t, t, y) - \varepsilon_\theta(x_t, t, \varnothing)\big)$$

$w = 0$ is unconditional, $w = 1$ is normal conditional, and $w > 1$ **exaggerates** the conditioning
— sharper, more prototypical, more obedient samples, at the cost of diversity. This single knob is
the "guidance scale" / "CFG" slider in every text-to-image tool, and $w \approx 3$–8 is typical.

The cost is two forward passes per step instead of one.

## 8. Latent diffusion — how the real systems work

Pixel-space diffusion at 512×512 is brutally expensive. **Latent diffusion** (Stable Diffusion) does
the obvious thing:

1. Train an autoencoder to compress 512×512×3 → 64×64×4 (a 48× reduction).
2. Run the entire diffusion process in that **latent space**.
3. Decode the final latent back to pixels.

Every component is something you've now built: the encoder/decoder is chapter 6's architecture (with
an adversarial loss term from chapter 7 to keep the decoder sharp), the denoiser is this chapter's
time-conditioned U-Net, text conditioning enters via cross-attention on CLIP embeddings, and the
guidance slider is the $w$ above.

That's the whole system. Nothing in it is beyond this course.

---

## Checklist before moving on

- [ ] Write the closed-form $q(x_t \mid x_0)$ from memory. What are $\alpha$, $\bar\alpha$?
- [ ] What exactly does the network predict, and what is the loss?
- [ ] Why predict $\varepsilon$ rather than $x_0$?
- [ ] Why sample a random $t$ per image rather than per batch?
- [ ] Why is noise added back during sampling? What happens without it?
- [ ] Why does the network need the timestep at all, and why sinusoidal features?
- [ ] What does DDIM change, and what does it cost?
- [ ] What does a guidance scale of 5 do, versus 1?
- [ ] Why is diffusion training stable when GAN training isn't?

*(Answers: $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\varepsilon$, with
$\alpha_t = 1-\beta_t$ and $\bar\alpha_t = \prod \alpha_s$; the added noise $\varepsilon$, with MSE;
constant target scale across all $t$, so no noise level dominates the loss; per-image gives far less
gradient variance since each step covers many noise levels; without injected noise the sampler
converges to the data mean — the MSE-blur failure; because the correct amount to remove depends
entirely on $t$, and a bare scalar is too weak a signal for 1000 distinguishable levels; DDIM makes
the reverse process deterministic so steps can be skipped — ~20× faster, slight quality cost, and
seeds become reproducible; it extrapolates away from the unconditional prediction, making samples
more prototypical and obedient but less diverse; because the target is known and fixed rather than
produced by an adversary that is itself moving.)*

---

## You've finished the course

Eight chapters, from `np.array` to a working diffusion model. What you can now do:

- Read any tensor shape and know what every axis means.
- Write a training loop from memory and debug one that's silently broken.
- Design a conv architecture on paper and predict its shapes and parameter count.
- Fine-tune a pretrained model correctly, and check *why* it's right.
- Build an encoder–decoder with skip connections and evaluate it honestly.
- Train both families of generative model, and explain the tradeoff between them.

Where to go next:

1. **Object detection** — the third core task, and the one this course skipped. Anchors, NMS, IoU
   matching, then a library (`torchvision` Faster R-CNN, or Ultralytics YOLO).
2. **Vision transformers** — attention instead of built-in locality, and why they need more data.
   Then CLIP, which is how text and images end up in one space.
3. **Self-supervised learning** — SimCLR, DINO, MAE. How modern backbones are actually pretrained.
4. **Fine-tuning diffusion** — LoRA, DreamBooth, ControlNet. Directly on top of this chapter, and
   the most immediately applicable thing on the list.
5. **Deployment** — `torch.compile`, ONNX, quantization, TensorRT. Different skills, and the ones
   that make your model useful to other people.

The habits matter more than any of it:

- **Look at your data and your predictions.** Plot them. Most bugs die on contact with a figure.
- **Overfit one batch** before you train for real. It separates bugs from tuning, every time.
- **Pick a metric that moves** when the thing you care about improves.
- **Ablate** to find out what actually helps, instead of guessing.
