import torch
import torch.nn.functional as F


def _create_gaussian_window_2d(
    win_size: int,
    sigma: float,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Creates a normalized 2D Gaussian kernel window for SSIM convolution.
    """
    coords = torch.arange(win_size, device=device, dtype=dtype) - (win_size - 1) / 2.0
    gauss_1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    gauss_2d = (gauss_1d.unsqueeze(1) @ gauss_1d.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    return gauss_2d.repeat(channels, 1, 1, 1)


def cosine_similarity_2d(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Computes mean Cosine Similarity between predicted and target 2D spectrograms.
    Flattens spatial dimensions per sample to measure directional spectral alignment.

    Args:
        pred: Predicted spectrogram tensor, shape (B, C, H, W) or (B, H, W).
        target: Ground-truth spectrogram tensor, shape (B, C, H, W) or (B, H, W).
        eps: Small epsilon for numerical stability.

    Returns:
        Scalar Tensor with mean cosine similarity across batch (in [-1.0, 1.0]).
    """
    p_flat = pred.flatten(1)
    t_flat = target.flatten(1)
    cos_sim = F.cosine_similarity(p_flat, t_flat, dim=1, eps=eps)
    return cos_sim.mean()


def ssim_2d(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 21.3,
    win_size: int = 7,
    win_sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    """
    Computes Structural Similarity Index Measure (SSIM) between 2D spectrograms.

    Args:
        pred: Predicted spectrogram tensor, shape (B, C, H, W) or (B, H, W).
        target: Ground-truth spectrogram tensor, shape (B, C, H, W) or (B, H, W).
        data_range: Dynamic range L of the signal. Default is 21.3, corresponding
                    to FSD50K log-mel spectrogram values (range ~ [-13.8, +7.5]).
                    If None, dynamically inferred from target min/max.
        win_size: Size of Gaussian sliding window (default: 7).
        win_sigma: Standard deviation of Gaussian window (default: 1.5).
        k1, k2: SSIM stability constants (default: 0.01 and 0.03).

    Returns:
        Scalar Tensor with mean SSIM across batch (in [-1.0, 1.0], 1.0 is identical).
    """
    if pred.ndim == 3:
        pred = pred.unsqueeze(1)
    if target.ndim == 3:
        target = target.unsqueeze(1)

    B, C, H, W = pred.shape
    device = pred.device
    dtype = pred.dtype

    if data_range is None:
        data_range = float((target.max() - target.min()).clamp(min=1e-6).item())

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    window = _create_gaussian_window_2d(win_size, win_sigma, C, device=device, dtype=dtype)
    pad = win_size // 2

    mu1 = F.conv2d(pred, window, padding=pad, groups=C)
    mu2 = F.conv2d(target, window, padding=pad, groups=C)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=pad, groups=C) - mu1_mu2

    numerator = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / denominator.clamp(min=1e-8)

    return ssim_map.mean()


class ReconstructionMetrics:
    """
    Accumulator for batch-wise evaluation of audio spectrogram reconstruction.
    Tracks:
      - MAE (Mean Absolute Error, L1)
      - MSE (Mean Squared Error, L2)
      - Cosine Similarity (directional spectral alignment)
      - SSIM (Structural Similarity Index Measure)
    """
    def __init__(self, data_range: float = 21.3, win_size: int = 7):
        self.data_range = data_range
        self.win_size = win_size
        self.reset()

    def reset(self):
        self.total_samples = 0
        self.sum_l1 = 0.0
        self.sum_mse = 0.0
        self.sum_cos_sim = 0.0
        self.sum_ssim = 0.0

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        batch_size = pred.shape[0]
        self.total_samples += batch_size

        l1 = F.l1_loss(pred, target, reduction="sum").item()
        mse = F.mse_loss(pred, target, reduction="sum").item()
        cos_sim = cosine_similarity_2d(pred, target).item() * batch_size
        ssim = ssim_2d(pred, target, data_range=self.data_range, win_size=self.win_size).item() * batch_size

        # Normalize L1 and MSE per element
        num_elements = pred.numel()
        elements_per_sample = num_elements / batch_size
        self.sum_l1 += l1 / elements_per_sample
        self.sum_mse += mse / elements_per_sample
        self.sum_cos_sim += cos_sim
        self.sum_ssim += ssim

    def compute(self) -> dict:
        if self.total_samples == 0:
            return {"mae": 0.0, "mse": 0.0, "cosine_similarity": 0.0, "ssim": 0.0}

        return {
            "mae": self.sum_l1 / self.total_samples,
            "mse": self.sum_mse / self.total_samples,
            "cosine_similarity": self.sum_cos_sim / self.total_samples,
            "ssim": self.sum_ssim / self.total_samples,
        }
