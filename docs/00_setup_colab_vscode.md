# Chapter 0 — Setup: no GPU here, so let Colab do the work

Your machine has no CUDA GPU. Chapters 1–3 don't care (NumPy, pandas, hand-written
convolutions — all CPU work). Chapters 4–6 train real CNNs, and on a CPU that is the
difference between 90 seconds and 45 minutes per epoch. So: **edit wherever you like,
train on Colab's free T4.**

One thing to be clear about up front, because it saves you an afternoon:

> **Free Colab has no official "attach VS Code to my Colab runtime" feature.**
> There is no first-party Google extension that turns VS Code into a Colab client.
> Anything claiming to do it is either (a) a tunnel hack, or (b) Colab **Enterprise**
> on Google Cloud, which is a paid product with a different UI.

So pick one of the three workflows below. Option A is what I recommend, and what this
repo is laid out for.

---

## Option A — GitHub as the sync layer (recommended)

Edit in VS Code, commit, open in Colab. Colab reads notebooks straight from GitHub and
can save back to it.

### One-time setup

```bash
cd C:\Users\1112582\PycharmProjects\ML_claudebook
git init
git add .
git commit -m "CV bootcamp"
```

Create an empty repo on GitHub, then:

```bash
git remote add origin https://github.com/<you>/ML_claudebook.git
git branch -M main
git push -u origin main
```

### Daily loop

1. In Colab: **File → Open notebook → GitHub tab** → paste your repo URL → pick a notebook.
2. Train. When you want to keep notebook changes: **File → Save a copy in GitHub**
   (it commits straight back to your branch).
3. Back in VS Code: `git pull`.

### Make it one click

Any notebook on GitHub opens in Colab by URL-swapping the domain:

```
https://github.com/<you>/ML_claudebook/blob/main/notebooks/04_cnn_classification.ipynb
->
https://colab.research.google.com/github/<you>/ML_claudebook/blob/main/notebooks/04_cnn_classification.ipynb
```

Drop this badge at the top of a notebook's first markdown cell and you get a button:

```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<you>/ML_claudebook/blob/main/notebooks/04_cnn_classification.ipynb)
```

The VS Code marketplace also has community "Open in Colab" extensions that do the URL
swap for the file you have open — search the Extensions pane for *Open in Colab*. Handy,
but it is only doing the string substitution above.

### Pull the whole repo into a Colab session

Some notebooks are nicer with `src/cv_utils.py` importable. First cell:

```python
!git clone -q https://github.com/<you>/ML_claudebook.git /content/ML_claudebook
import sys; sys.path.append('/content/ML_claudebook/src')
from cv_utils import set_seed, get_device
```

(The course notebooks don't *require* this — they redefine what they need so they work
standalone.)

---

## Option B — Google Drive as the sync layer

Good if you don't want to use git for scratch work.

1. Install **Google Drive for desktop** and sign in. You get a `G:\My Drive` (letter varies).
2. Put the project in Drive, e.g. `G:\My Drive\ML_claudebook`.
3. Open VS Code on that folder — you are editing files that sync to Drive automatically.
4. In Colab, mount Drive and open the notebook from the file browser:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/ML_claudebook
```

Caveats: Drive sync + Jupyter both writing the same `.ipynb` can produce conflicted
copies, and Drive I/O inside Colab is slow — never point a `DataLoader` at Drive for
training data. Copy datasets to `/content/` first (local SSD), and use Drive only for
checkpoints:

```python
CKPT_DIR = '/content/drive/MyDrive/cv_bootcamp_ckpt'   # survives runtime restarts
DATA_DIR = '/content/data'                             # fast, wiped on disconnect
```

---

## Option C — VS Code attached to a Colab runtime over SSH (advanced, fragile)

Only if you genuinely need the VS Code debugger on the GPU box. This tunnels a shell out
of the Colab VM and attaches VS Code's Remote-SSH to it. It breaks whenever Colab changes
its sandbox, and Colab's terms discourage using the runtime as a general remote server —
don't lean on it for anything important.

In a Colab cell:

```python
!pip install -q colab-ssh
from colab_ssh import launch_ssh_cloudflared
launch_ssh_cloudflared(password='pick-something')
```

It prints a hostname + the VS Code Remote-SSH config block to paste into `~/.ssh/config`.
Then in VS Code: **Remote-SSH: Connect to Host**.

A cleaner variant of the same idea: run a **VS Code tunnel** from the Colab VM, then open
it at <https://vscode.dev> or attach from desktop VS Code (`Remote - Tunnels` extension):

```python
!curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' \
  --output vscode_cli.tar.gz && tar -xf vscode_cli.tar.gz
!./code tunnel --accept-server-license-terms
```

It prints a device-login code. Either way: the runtime dies after ~12 hours (or ~90
minutes idle) and takes your tunnel with it.

---

## Whichever option you pick: verify the GPU first

Every notebook in chapters 4–6 opens with this cell. Run it before anything else.

```python
import sys, platform
print('python  ', sys.version.split()[0], platform.system())

try:
    import google.colab          # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
print('in colab', IN_COLAB)

import torch, torchvision
print('torch   ', torch.__version__, '| torchvision', torchvision.__version__)
print('cuda ok ', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu     ', torch.cuda.get_device_name(0),
          f'{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
else:
    print('NO GPU -> Runtime > Change runtime type > T4 GPU, then Runtime > Restart')
```

Expected on free Colab: `Tesla T4  15.8 GB`.

`!nvidia-smi` gives you the same thing plus live memory use — useful when you hit
`CUDA out of memory` and want to know who ate the VRAM.

### Colab housekeeping that matters

| Thing | Why you care |
|---|---|
| **Runtime → Change runtime type → T4 GPU** | Notebooks in this repo request GPU in metadata, but Colab still asks. |
| **Runtime → Disconnect and delete runtime** | The real "restart from clean" — frees VRAM leaked by a crashed cell. |
| ~90 min idle / ~12 h max session | Save checkpoints to Drive or you *will* lose a training run. |
| `/content` is wiped on disconnect | Datasets are re-downloaded next session. That's fine; they're small here. |
| `%pip install` not `!pip install` | `%pip` installs into the kernel's env. Restart the runtime after installing anything that touches torch. |
| Free tier has no guaranteed GPU | If you get "no backend available", try again later or use CPU for chapters 1–3. |

### Keeping local VS Code useful without a GPU

- Install the **Jupyter** (`ms-toolsai.jupyter`) and **Python** extensions. You can open
  and read every `.ipynb` here locally, run chapters 1–3, and read the outputs Colab saved.
- Local dev install (CPU wheels — ~200 MB instead of ~2.5 GB):

  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```

- Write and debug model *code* locally against a tiny fake batch — this catches
  90% of bugs without a GPU:

  ```python
  x = torch.randn(2, 3, 32, 32)      # 2 fake images
  print(model(x).shape)              # shape bugs surface instantly
  ```

  Then push and train on Colab. Chapter 4 formalises this as the
  *overfit-one-batch* sanity check.

---

Next: [Chapter 1 — NumPy & pandas for image data](01_numpy_pandas.md)
