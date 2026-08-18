# External resources

## Summary — the short answer

| Question | Answer |
|---|---|
| External **datasets** used? | **None.** Organizer-provided `train/` pairs only. |
| Pretrained weights in the **submitted model**? | **None.** Trained from random initialization. |
| Pretrained weights anywhere in the project? | **Yes — evaluation only.** LPIPS + its AlexNet backbone, used solely to compute a reported metric. Not imported by `inference.py`. |
| Licences verified? | **Yes**, read from the `LICENSE` file shipped in each installed package. |

`inference.py` — the graded entry point — imports **no** package that fetches
anything from the network. This was verified by tracing its full transitive import
graph (1,108 modules; `lpips`, `torchvision` and `skimage` all absent) and then
running it with those packages made unimportable *and* outbound sockets disabled:
it completed all 400 test images and produced byte-identical outputs. The graded run
is safe in an offline or sandboxed environment.

---

Every third-party model, pretrained weight, and dataset used anywhere in this
project. Licences below were read from the actual `LICENSE` file shipped in the
installed package (path given per entry), not quoted from memory.

## Pretrained model weights used in the **submitted restoration model**

**None.** The submitted model (`src/models/nafnet_sr.py`) is trained from random
initialization on the organizer-provided `train/` pairs only. No pretrained
backbone, no external checkpoint, no GAN-pretrained initialization.

## Pretrained weights used for **evaluation only** (not part of inference)

### 1. LPIPS (Learned Perceptual Image Patch Similarity)

| | |
|---|---|
| Used for | Computing the LPIPS metric in `scripts/evaluate.py`. **Not** used in the loss, not used in `inference.py`, not part of the submitted model. |
| Package | `lpips==0.1.4` (PyPI) |
| Source repo | https://github.com/richzhang/PerceptualSimilarity |
| Licence | **BSD 2-Clause**, read from `site-packages/lpips-0.1.4.dist-info/LICENSE` — "Copyright (c) 2018, Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang. All rights reserved." Package classifier: `License :: OSI Approved :: BSD License`. |
| Paper | Zhang, Isola, Efros, Shechtman, Wang. *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*. CVPR 2018. https://arxiv.org/abs/1801.03924 |
| Variant | `net='alex'` (the repo's default and the variant reported in the paper's benchmarks) |

### 2. AlexNet ImageNet weights (a dependency of LPIPS above)

| | |
|---|---|
| Used for | Backbone that LPIPS builds its perceptual distance on. Downloaded automatically by `lpips` on first use to `~/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth` (233 MB). |
| Package | `torchvision==0.27.0` |
| Licence | **BSD 3-Clause**, read from `site-packages/torchvision-0.27.0+cu126.dist-info/LICENSE` — "Copyright (c) Soumith Chintala 2016". Distribution metadata `License: BSD`. |
| Model card | https://pytorch.org/vision/stable/models/generated/torchvision.models.alexnet.html |
| Paper | Krizhevsky, Sutskever, Hinton. *ImageNet Classification with Deep Convolutional Neural Networks*. NeurIPS 2012. |

> Note: this weight is reached only through the LPIPS metric. Deleting
> `scripts/evaluate.py` would remove the project's only dependence on any
> pretrained weight.

## Datasets

**No external datasets were used.** Training, validation, and all reported metrics
come exclusively from the organizer-provided `train/` pairs. The held-out
validation split is a seeded subset of `train/`, listed verbatim in
`configs/val_split.txt`.

The organizer-provided `Test_NoisyLR/` inputs were used **only** to produce final
restored outputs for submission — never for training, never for validation, never
for checkpoint selection.

## Architecture provenance (code written here, no weights imported)

The model is a NAFNet-style encoder–decoder with a PixelShuffle ×2 head. The
NAFBlock design (SimpleGate + simplified channel attention, no LayerNorm-free
claims beyond what is implemented) follows:

> Chen, Chu, Zhang, Sun. *Simple Baselines for Image Restoration*. ECCV 2022.
> https://arxiv.org/abs/2204.04676 — reference implementation
> https://github.com/megvii-research/NAFNet (MIT licence).

Our `src/models/nafnet_sr.py` is an independent implementation written for this
project. No code and no weights were copied from that repository; it is cited as
the source of the architectural idea.
