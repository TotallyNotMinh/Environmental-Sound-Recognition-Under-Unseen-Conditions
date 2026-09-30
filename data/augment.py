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
        if freq_mask_param > 0 and num_masks > 0:
            self.masker = T.FrequencyMasking(freq_mask_param=freq_mask_param, iid_masks=iid_masks)
        else:
            self.masker = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.masker is None:
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
        if time_mask_param > 0 and num_masks > 0:
            self.masker = T.TimeMasking(time_mask_param=time_mask_param, iid_masks=iid_masks)
        else:
            self.masker = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.masker is None:
            return x
        for _ in range(self.num_masks):
            x = self.masker(x)
        return x


class SpecAugment(nn.Module):
    """
    SpecAugment module combining frequency and time masking for audio spectrograms.
    AST / CMKD defaults for FSD50K:
      - Frequency mask param: 48 (2 masks)
      - Time mask param: 192 (2 masks)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.time_mask(self.freq_mask(x))


class RandomTimeShift(nn.Module):
    """
    Random circular time shift along the time axis (last dimension).
    FSD50K / AST standard: +/- 10 frames.
    """
    def __init__(self, max_shift=10):
        super().__init__()
        self.max_shift = max_shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_shift <= 0:
            return x
        shift = random.randint(-self.max_shift, self.max_shift)
        if shift == 0:
            return x
        return torch.roll(x, shifts=shift, dims=-1)


class RandomNoise(nn.Module):
    """
    Uniform additive random noise on spectrogram bins.
    CMKD paper standard for FSD50K: U(0, max_noise) with max_noise = 0.05.
    """
    def __init__(self, max_noise=0.05):
        super().__init__()
        self.max_noise = max_noise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_noise <= 0.0:
            return x
        noise = torch.rand_like(x) * self.max_noise
        return x + noise


def mixup_samples(
    x1: torch.Tensor,
    y1: torch.Tensor,
    x2: torch.Tensor,
    y2: torch.Tensor,
    alpha: float = 0.5,
    label_mode: str = "union",
    lam: float = None,
    return_lam: bool = False,
):
    """
    Mixup between two samples (or dual view representations).

    Parameters:
      x1, x2: Input audio spectrograms of shape (1, F, T) or (F, T).
      y1, y2: Target label vectors or multi-hot binary vectors.
      alpha: Beta distribution parameter Beta(alpha, alpha).
      label_mode:
        - 'union': y_mix = max(y1, y2). Multi-label union preserving all active
                   sound events without fractional suppression penalties.
        - 'linear': y_mix = lam * y1 + (1 - lam) * y2. Traditional linear interpolation.
      lam: If provided, reuses existing lambda (crucial for synchronizing dual clean/noisy views).
      return_lam: If True, returns (mixed_x, mixed_y, lam). If False, returns (mixed_x, mixed_y).

    Returns:
      (mixed_x, mixed_y) or (mixed_x, mixed_y, lam)
    """
    if alpha <= 0.0:
        return (x1, y1, 1.0) if return_lam else (x1, y1)

    if lam is None:
        lam = float(np.random.beta(alpha, alpha))

    mixed_x = lam * x1 + (1.0 - lam) * x2

    if label_mode == "union" and isinstance(y1, torch.Tensor) and isinstance(y2, torch.Tensor) and y1.ndim in (1, 2):
        mixed_y = torch.maximum(y1, y2)
    else:
        mixed_y = lam * y1 + (1.0 - lam) * y2

    if return_lam:
        return mixed_x, mixed_y, lam
    return mixed_x, mixed_y


def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.5,
    label_mode: str = "union",
):
    """
    Mixup applied across a mini-batch along dimension 0.

    Parameters:
      x: Tensor of shape (B, C, F, T)
      y: Tensor of shape (B, num_classes)
      alpha: Beta distribution parameter
      label_mode: 'union' or 'linear'

    Returns:
      mixed_x, mixed_y, lam
    """
    if alpha <= 0.0 or x.size(0) <= 1:
        return x, y, 1.0

    lam = float(np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    perm = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1.0 - lam) * x[perm]

    if label_mode == "union" and isinstance(y, torch.Tensor) and y.ndim in (1, 2):
        mixed_y = torch.maximum(y, y[perm])
    else:
        mixed_y = lam * y + (1.0 - lam) * y[perm]

    return mixed_x, mixed_y, lam


class Mixup:
    """
    Callable Mixup wrapper supporting both sample-level and batch-level invocation.
    """
    def __init__(self, alpha: float = 0.5, p: float = 0.5, label_mode: str = "union"):
        self.alpha = alpha
        self.p = p
        self.label_mode = label_mode

    def __call__(self, x1, y1, x2=None, y2=None):
        if random.random() > self.p:
            return (x1, y1)

        if x2 is None:
            mixed_x, mixed_y, _ = mixup_batch(x1, y1, alpha=self.alpha, label_mode=self.label_mode)
            return mixed_x, mixed_y

        return mixup_samples(x1, y1, x2, y2, alpha=self.alpha, label_mode=self.label_mode, return_lam=False)
