# CV Bootcamp — from NumPy to Diffusion

A self-paced refresher course in classical + deep-learning computer vision, taught in
PyTorch. Eight chapters, each with a **written lesson**, a **worked notebook**, and a
**graded exercise notebook** (solutions included).

Designed for a machine **without a GPU**: chapters 1–3 run fine on a CPU, chapters 4–8
expect a free Google Colab T4. See [docs/00_setup_colab_vscode.md](docs/00_setup_colab_vscode.md)
for the VS Code ↔ Colab workflow.

---

## Curriculum

| # | Topic | Lesson | Worked notebook | Exercise | GPU |
|---|-------|--------|-----------------|----------|-----|
| 1 | NumPy & pandas for image data | [docs](docs/01_numpy_pandas.md) | [nb](notebooks/01_numpy_pandas.ipynb) | [ex](exercises/ex01_numpy_pandas.ipynb) · [sol](exercises/solutions/sol01_numpy_pandas.ipynb) | no |
| 2 | Linear & logistic regression, gradient descent, autograd | [docs](docs/02_ml_foundations.md) | [nb](notebooks/02_linear_logistic_regression.ipynb) | [ex](exercises/ex02_regression.ipynb) · [sol](exercises/solutions/sol02_regression.ipynb) | no |
| 3 | Convolution, filters, pooling, receptive fields | [docs](docs/03_convolution.md) | [nb](notebooks/03_convolution_image_ops.ipynb) | [ex](exercises/ex03_convolution.ipynb) · [sol](exercises/solutions/sol03_convolution.ipynb) | no |
| 4 | CNN image classification (CIFAR-10), full training loop | [docs](docs/04_cnn_classification.md) | [nb](notebooks/04_cnn_classification.ipynb) | [ex](exercises/ex04_cnn.ipynb) · [sol](exercises/solutions/sol04_cnn.ipynb) | yes |
| 5 | Transfer learning & fine-tuning a pretrained ResNet | [docs](docs/05_transfer_learning.md) | [nb](notebooks/05_transfer_learning.ipynb) | [ex](exercises/ex05_transfer.ipynb) · [sol](exercises/solutions/sol05_transfer.ipynb) | yes |
| 6 | Semantic segmentation with a U-Net | [docs](docs/06_segmentation.md) | [nb](notebooks/06_segmentation_unet.ipynb) | [ex](exercises/ex06_segmentation.ipynb) · [sol](exercises/solutions/sol06_segmentation.ipynb) | yes |
| 7 | GANs: adversarial generation, DCGAN, conditional GAN | [docs](docs/07_gans.md) | [nb](notebooks/07_gan_dcgan.ipynb) | [ex](exercises/ex07_gan.ipynb) · [sol](exercises/solutions/sol07_gan.ipynb) | yes |
| 8 | Diffusion: DDPM from scratch, DDIM, guidance | [docs](docs/08_diffusion.md) | [nb](notebooks/08_diffusion_ddpm.ipynb) | [ex](exercises/ex08_diffusion.ipynb) · [sol](exercises/solutions/sol08_diffusion.ipynb) | yes |

Chapters 1–6 build a *discriminative* pipeline (image → label / mask). Chapters 7–8 turn it
around: **generative** modelling, where there is no ground-truth output and defining the
objective is the whole problem. Chapter 8 reuses the U-Net from chapter 6 directly — the
architecture you wrote for segmentation is the denoising backbone of a diffusion model.

Reference material you will come back to:

- [docs/90_pytorch_cheatsheet.md](docs/90_pytorch_cheatsheet.md) — the API surface you actually use daily.
- [docs/91_debugging_playbook.md](docs/91_debugging_playbook.md) — "loss is NaN", "accuracy stuck at 10%", shape errors, and the rest.

## How to work through a chapter

1. **Read** the lesson in `docs/`. It is the theory, the shape algebra, and the pitfalls.
2. **Run** the matching notebook in `notebooks/`, cell by cell. Change numbers and re-run —
   the notebooks are written to be poked at, and most sections end with a "try this" nudge.
3. **Do** the exercise in `exercises/` without looking at the solution. Each task has a
   `TODO` and an assertion or a plot that tells you whether you got it right.
4. **Compare** with `exercises/solutions/`. The solutions explain *why*, not just *what*.

Budget roughly 2–4 hours per chapter. Chapters build on each other; chapter 6 assumes
you are comfortable with the training loop from chapter 4, and chapter 8 assumes chapter 6's
U-Net and chapter 7's conditioning trick.

## Quick start (Colab, recommended)

1. Push this folder to a GitHub repo (see [docs/00_setup_colab_vscode.md](docs/00_setup_colab_vscode.md)).
2. Open <https://colab.research.google.com> → **GitHub** tab → pick a notebook.
3. For chapters 4–8: **Runtime → Change runtime type → T4 GPU**.
4. Run the first cell of any notebook — it prints your environment and GPU, and installs
   anything missing.

## Quick start (local CPU, chapters 1–3)

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Note: this machine currently has no real Python install (only the Windows Store stub),
so the local path needs Python 3.10+ installed first — or just use Colab for everything.

## Repo layout

```
docs/         written lessons + reference sheets (read these first)
notebooks/    worked example notebooks, one per chapter
exercises/    exercise notebooks with TODOs; solutions/ has the answers
src/          cv_utils.py - small helpers for your own local experiments
tools/        how the notebooks are generated (see below)
```

### About `tools/`

Notebook JSON is horrible to review in a diff, so every notebook is authored as a plain
text file in `tools/nbsrc/*.nbsrc` and compiled by a PowerShell script. You never need to
run this — the `.ipynb` files are committed. But if you want to edit a lesson as text:

```bash
powershell -ExecutionPolicy Bypass -File tools/build_notebooks.ps1
```

And to re-check structure and links afterwards:

```bash
powershell -ExecutionPolicy Bypass -File tools/validate.ps1
```

Edit the `.ipynb` directly if you prefer; just don't do both for the same notebook and
expect them to stay in sync.

### Status: cells are unexecuted

Every notebook ships with empty outputs. They were written and structurally validated on a
machine with no Python interpreter, so **the code has not been run yet** — Colab is the first
execution. If a cell errors, it's a genuine bug and not a deliberate exercise; the exercise
notebooks mark their gaps explicitly with `TODO` and `raise NotImplementedError`.

The most likely places for version friction, if you hit any:

- `matplotlib.cbook.get_sample_data('grace_hopper.jpg')` in chapter 3 (there's a synthetic
  fallback built in if the sample image isn't available).
- `torch.amp.GradScaler` vs `torch.cuda.amp.GradScaler` — chapters 4 and 6 try both.
- `torchvision.transforms.v2` — chapter 3 falls back to v1.
- The FID cell in chapter 7 downloads Inception-v3 and spends most of its time in
  `scipy.linalg.sqrtm`; it's the slowest cell in the course (~1–2 min).

Rough GPU runtimes on a free Colab T4: chapters 4 and 6 about 8 minutes each, chapter 5 about 4,
chapter 7 about 10 (two GANs plus FID), chapter 8 about 8 (two diffusion models plus sampling).
