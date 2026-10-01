import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F


class KDLoss(nn.Module):
    """
    CMKD knowledge-distillation loss for multi-label audio classification
    (docs/cmkd-paper-notes.md section 4):

        Loss = lam * BCE(z_s, y_smoothed) + (1 - lam) * KL(sigmoid(z_t / tau) || sigmoid(z_s))

    - sigmoid (not softmax) because FSD50K is multi-label, so the KL term is the
      per-class Bernoulli KL.
    - tau is applied to the TEACHER logits only, with no tau^2 rescale (CMKD's
      deliberate departure from classic Hinton KD).
    - Both terms are averaged over batch AND classes, so they share the same scale
      as nn.BCEWithLogitsLoss's default reduction and lam balances them as intended.
    - Label smoothing applies to the ground-truth term only, with the same formula
      scripts/train_cnn_classifier.py uses.
    - Computed in float32 regardless of autocast, since log-sigmoid differences are
      precision-sensitive.
    """
    def __init__(self, lam=0.5, tau=1.0, label_smoothing=0.1):
        super().__init__()
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f"lam must be in [0, 1], got {lam}")
        if tau <= 0.0:
            raise ValueError(f"tau must be > 0, got {tau}")
        self.lam = lam
        self.tau = tau
        self.label_smoothing = label_smoothing

    def forward(self, student_logits, teacher_logits, targets):
        z_s = student_logits.float()
        z_t = teacher_logits.detach().float() / self.tau
        targets = targets.float()

        if self.label_smoothing > 0.0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        gt_loss = F.binary_cross_entropy_with_logits(z_s, targets)

        # Bernoulli KL(p_t || p_s), via log-sigmoid for numerical stability:
        # log(p) = logsigmoid(z), log(1 - p) = logsigmoid(-z)
        p_t = torch.sigmoid(z_t)
        kl = (
            p_t * (F.logsigmoid(z_t) - F.logsigmoid(z_s))
            + (1.0 - p_t) * (F.logsigmoid(-z_t) - F.logsigmoid(-z_s))
        )
        kd_loss = kl.mean()

        total = self.lam * gt_loss + (1.0 - self.lam) * kd_loss
        return total, gt_loss, kd_loss


if __name__ == "__main__":
    torch.manual_seed(0)
    B, C = 4, 200
    targets = (torch.rand(B, C) > 0.95).float()
    z = torch.randn(B, C) * 3

    # 1. Identical student/teacher at tau=1 -> KD term ~ 0
    _, _, kd = KDLoss(tau=1.0)(z, z, targets)
    assert kd.abs() < 1e-6, f"KD term should be ~0 for identical logits, got {kd.item()}"

    # 2. lam=1 -> total equals plain smoothed BCE
    total, gt, _ = KDLoss(lam=1.0)(z, torch.randn(B, C), targets)
    smoothed = targets * 0.9 + 0.05
    assert torch.allclose(total, F.binary_cross_entropy_with_logits(z, smoothed)), "lam=1 should reduce to BCE"

    # 3. Finite for extreme logits
    big_s = torch.full((B, C), 50.0)
    big_t = torch.full((B, C), -50.0)
    total, gt, kd = KDLoss()(big_s, big_t, targets)
    assert torch.isfinite(total) and torch.isfinite(kd), "Loss must be finite for +/-50 logits"

    # 4. KL is non-negative
    _, _, kd = KDLoss()(torch.randn(B, C), torch.randn(B, C), targets)
    assert kd >= 0, f"KL must be >= 0, got {kd.item()}"

    print("KDLoss sanity checks passed.")
