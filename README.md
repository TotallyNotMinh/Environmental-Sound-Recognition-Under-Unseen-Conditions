# Environmental Sound Recognition Under Unseen Conditions (ESRUUC-CMKD)

Research pipeline for environmental sound / audio-event classification that stays robust when the recording conditions at test time weren't seen during training. Built around an Audio Spectrogram Transformer (AST), self-supervised denoiser pretraining, and (in progress) CNN↔Transformer Cross-Model Knowledge Distillation (CMKD) to further close the generalization gap.

## Overview

The pipeline has three parts:

1. **Denoiser pretraining** — an AST-style encoder + convolutional decoder is pretrained on the ACAD dataset to reconstruct clean background audio from noisy mixtures. This is a self-supervised warm-start for the encoder before classification training.
2. **AST classifier fine-tuning** — the same encoder plus a linear head is fine-tuned on FSD50K for 200-class multi-label sound-event classification, optionally warm-started from the pretrained denoiser.
3. **CMKD CNN teacher** *(in progress)* — an EfficientNet-B0 CNN classifier, trained on FSD50K with the exact recipe from the CMKD paper, intended as the frozen teacher for a future CNN→AST-Base cross-model distillation step. CMKD's own findings show this direction specifically transfers a CNN's translation-invariance into the AST student — the property this project cares about, since robustness to shifted/unseen recording conditions is the actual research question.

Generalization is evaluated against a dedicated held-out benchmark of unseen recording conditions: [kaggle.com/datasets/minhtom/audio-evaluation-benchmark](https://www.kaggle.com/datasets/minhtom/audio-evaluation-benchmark).

## Project layout

```
models/
  encoder.py          AST encoder (DeiT/DINO-pretrained ViT, 16x16 overlapping patches)
  decoder.py           Convolutional decoder for denoiser pretraining
  denoiser.py          Encoder + decoder, trained on ACAD
  classifier.py         Encoder + linear head, trained on FSD50K
  cnn_classifier.py    EfficientNet-based CNN classifier (CMKD teacher)
losses/
  distillation.py      CMKD KD loss (BCE + per-class Bernoulli KL, teacher temperature)
data/
  dataset.py           ACADDataset, FSD50KDataset
  augment.py           SpecAugment, mixup, random noise/time-shift
  sampler.py           Class-balanced distributed sampler for multi-GPU training
metrics/
  classification.py   mAP, mAUC, micro/macro-F1, top-1/top-5 hit rate
scripts/
  train_denoise.py           Denoiser pretraining (ACAD)
  train_classifier.py        AST classifier fine-tuning (FSD50K)
  train_cnn_classifier.py    CNN teacher training (FSD50K)
  train_kd.py                CNN->AST knowledge distillation (FSD50K)
  evaluate.py                Score a CNN/AST checkpoint on the FSD50K eval (or val) split
  aws/                       One-command launchers for AWS g5/A10G (see docs/aws-a10g-guide.md)
docs/
  cmkd-paper-notes.md  Research notes on the CMKD/AST/PSLA papers this project builds on
train.ipynb            Kaggle launch notebook (clone, install, torchrun each script)
```

## Setup

No lockfile is tracked — dependencies (`torch`, `torchaudio`, `torchvision`, `tqdm`, `torchinfo` (optional, for model summaries)) are installed manually. Kaggle's default PyTorch Docker image already bundles all of these, and training is designed to run there (see `train.ipynb`) rather than locally.

## Quickstart

Every script supports `--mock`, which swaps in synthetic tensors in place of real dataset files — useful for sanity-checking a change without needing ACAD or FSD50K on disk:

```bash
python scripts/train_denoise.py --mock
python scripts/train_classifier.py --mock
python scripts/train_cnn_classifier.py --mock
python scripts/train_kd.py --mock            # uses a random teacher when --teacher-checkpoint is omitted
```

## Training on Kaggle

`train.ipynb` is the actual multi-GPU launch reference: it clones the repo, installs a few extra dependencies, then runs each training script via `torchrun`. All scripts auto-detect `RANK`/`WORLD_SIZE` env vars, so the same script runs single- or multi-GPU without a flag.

```bash
torchrun --nproc_per_node=2 scripts/train_classifier.py --data-path data/fsd50k --batch-size 48
torchrun --nproc_per_node=2 scripts/train_cnn_classifier.py --data-path data/fsd50k --batch-size 24
torchrun --nproc_per_node=2 scripts/train_kd.py --data-path data/fsd50k --batch-size 12 \
        --teacher-checkpoint checkpoints/fsd50k_cnn_b0/best_cnn.pth
```

Note: `train_classifier.py` takes a per-GPU `--batch-size`; the two CMKD scripts take the **total** batch and split it across GPUs.

## Training on AWS (A10G)

Kaggle's 12 h session limit is too short for the 50-epoch CMKD runs. `docs/aws-a10g-guide.md` covers
instance choice, setup and data download. After that, the experiment is:

```bash
bash scripts/aws/quick_check.sh
bash scripts/aws/run_cnn.sh     # STEP 1: CNN teacher
bash scripts/aws/run_kd.sh      # STEP 2: KD CNN -> AST-Base
```

## Datasets

- **ACAD** (Automatic Contextual Audio Denoising benchmark, Luong et al., EUSIPCO 2026) — paired noisy/clean scene recordings (Kitchen, Park, Restaurant, Restroom, Street, Subway), used for denoiser pretraining.
- **FSD50K** — a 200-class multi-label sound-event dataset (Kaggle mirror: `yousirui1/fsd50k`), used for classifier fine-tuning and for training the CNN teacher.
- **Unseen-condition evaluation benchmark** (custom) — [kaggle.com/datasets/minhtom/audio-evaluation-benchmark](https://www.kaggle.com/datasets/minhtom/audio-evaluation-benchmark), held out specifically for testing generalization to recording conditions not seen during training.

## Roadmap

- [x] AST denoiser pretraining + classifier fine-tuning pipeline
- [x] CMKD-style EfficientNet-B0 CNN teacher, trained standalone on FSD50K
- [x] CNN→AST-Base cross-model knowledge distillation training loop (`scripts/train_kd.py`)
- [ ] Run the KD experiment on Kaggle and compare against the paper's 61.7 mAP
- [ ] Evaluation of denoiser / AST classifier / CNN teacher / distilled model against the unseen-condition benchmark

## References

- Gong, Chung, Glass. *AST: Audio Spectrogram Transformer.* Interspeech 2021. [arXiv:2104.01778](https://arxiv.org/abs/2104.01778)
- Gong, Chung, Glass. *PSLA: Improving Audio Tagging with Pretraining, Sampling, Labeling, and Aggregation.* IEEE/ACM TASLP, 2021. [arXiv:2102.01243](https://arxiv.org/abs/2102.01243)
- Gong, Khurana, Rouditchenko, Glass. *CMKD: CNN/Transformer-Based Cross-Model Knowledge Distillation for Audio Classification.* 2022. [arXiv:2203.06760](https://arxiv.org/abs/2203.06760)

See `docs/cmkd-paper-notes.md` for detailed research notes on the CMKD paper, including the exact architecture and hyperparameters this project reproduces.
