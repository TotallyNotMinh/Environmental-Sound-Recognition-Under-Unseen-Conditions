import random
import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T


class FrequencyMasking(nn.Module):
    """
    Applies frequency masking to a spectrogram of shape (..., F, T).
    """
    def __init__(self, freq_mask_param=24, num_masks=1, iid_masks=True):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.num_masks = num_masks
        if freq_mask_param > 0:
            self.masker = T.FrequencyMasking(freq_mask_param=freq_mask_param, iid_masks=iid_masks)
        else:
            self.masker = None

    def forward(self, x):
        if self.masker is None or self.num_masks <= 0:
            return x
        for _ in range(self.num_masks):
            x = self.masker(x)
        return x


class TimeMasking(nn.Module):
    """
    Applies time masking to a spectrogram of shape (..., F, T).
    """
    def __init__(self, time_mask_param=48, num_masks=1, iid_masks=True):
        super().__init__()
        self.time_mask_param = time_mask_param
        self.num_masks = num_masks
        if time_mask_param > 0:
            self.masker = T.TimeMasking(time_mask_param=time_mask_param, iid_masks=iid_masks)
        else:
            self.masker = None

    def forward(self, x):
        if self.masker is None or self.num_masks <= 0:
            return x
        for _ in range(self.num_masks):
            x = self.masker(x)
        return x


class RandomTimeShift(nn.Module):
    """
    Random time shift augmentation along time axis.
    Paper: +/- 10 frames for FSD50K.
    """
    def __init__(self, max_shift=10):
        super().__init__()
        self.max_shift = max_shift

    def forward(self, x):
        if self.max_shift <= 0:
            return x
        shift = random.randint(-self.max_shift, self.max_shift)
        if shift == 0:
            return x
        return torch.roll(x, shifts=shift, dims=-1)


class RandomNoise(nn.Module):
    """
    Uniform additive random noise on spectrogram.
    Paper: U(0, 0.05) on spectrogram for FSD50K.
    """
    def __init__(self, max_noise=0.05):
        super().__init__()
        self.max_noise = max_noise

    def forward(self, x):
        if self.max_noise <= 0:
            return x
        noise = torch.rand_like(x) * self.max_noise
        return x + noise


class SpecAugment(nn.Module):
    """
    SpecAugment module combining frequency and time masking for audio spectrograms.
    Defaults matching AST / CMKD: freq_mask_param=48, time_mask_param=192.
    """
    def __init__(
        self,
        freq_mask_param=48,
        time_mask_param=192,
        num_freq_masks=2,
        num_time_masks=2,
        iid_masks=True,
    ):
        super().__init__()
        self.freq_mask = FrequencyMasking(
            freq_mask_param=freq_mask_param,
            num_masks=num_freq_masks,
            iid_masks=iid_masks,
        )
        self.time_mask = TimeMasking(
            time_mask_param=time_mask_param,
            num_masks=num_time_masks,
            iid_masks=iid_masks,
        )

    def forward(self, x):
        return self.time_mask(self.freq_mask(x))


def mixup_samples(x1, y1, x2, y2, alpha=0.5):
    """
    Mixup between two samples (or pairs of tensors).
    Works for:
      - Multi-label classification: x=spectrogram, y=binary/multi-hot vector
      - Denoising: x=noisy_spectrogram, y=clean_spectrogram
    Returns:
      mixed_x, mixed_y
    """
    if alpha <= 0.0:
        return x1, y1
    lam = float(np.random.beta(alpha, alpha))
    mixed_x = lam * x1 + (1.0 - lam) * x2
    mixed_y = lam * y1 + (1.0 - lam) * y2
    return mixed_x, mixed_y


def mixup_batch(x, y, alpha=0.5):
    """
    Mixup applied across a mini-batch along dimension 0.
    x: Tensor (B, ...)
    y: Tensor (B, ...)
    Returns:
      mixed_x, mixed_y, lam
    """
    if alpha <= 0.0 or x.size(0) <= 1:
        return x, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    perm = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[perm]
    mixed_y = lam * y + (1.0 - lam) * y[perm]
    return mixed_x, mixed_y, lam


class Mixup:
    """
    Mixup wrapper supporting both sample-level and batch-level invocation.
    """
    def __init__(self, alpha=0.5, p=0.5):
        self.alpha = alpha
        self.p = p

    def __call__(self, x1, y1, x2=None, y2=None):
        if random.random() > self.p:
            if x2 is None:
                return x1, y1
            return x1, y1
        if x2 is None:
            mixed_x, mixed_y, _ = mixup_batch(x1, y1, alpha=self.alpha)
            return mixed_x, mixed_y
        return mixup_samples(x1, y1, x2, y2, alpha=self.alpha)
