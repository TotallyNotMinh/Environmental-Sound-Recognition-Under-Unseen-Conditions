import math
import random
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T


class RealWorldNoiseBank:
    """
    In-memory bank of real-world environmental noise clips (e.g., TAU Urban Acoustic Scenes).
    Pre-caches clips into RAM at target_sample_rate to eliminate disk I/O bottleneck during training.
    """
    def __init__(
        self,
        noise_dir: str,
        target_sample_rate: int = 16000,
        bank_size: int = 1000,
        seed: int = 42,
    ):
        self.target_sample_rate = target_sample_rate
        noise_path = Path(noise_dir)
        if not noise_path.exists():
            raise FileNotFoundError(f"Noise directory does not exist: {noise_dir}")

        self.noise_files = list(noise_path.glob("*.wav"))
        if not self.noise_files:
            raise FileNotFoundError(f"No .wav files found in noise directory: {noise_dir}")

        rng = random.Random(seed)
        sampled_files = rng.sample(self.noise_files, min(bank_size, len(self.noise_files)))

        # Pre-cache resampled noise buffers into memory (1000 1-sec clips = ~64MB RAM)
        self.buffers = []
        resamplers = {}
        for p in sampled_files:
            try:
                waveform, sr = torchaudio.load(str(p))
                if waveform.ndim > 1 and waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                elif waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)

                if sr != target_sample_rate:
                    if sr not in resamplers:
                        resamplers[sr] = T.Resample(orig_freq=sr, new_freq=target_sample_rate)
                    waveform = resamplers[sr](waveform)

                self.buffers.append(waveform.squeeze(0).contiguous())
            except Exception:
                continue

        if not self.buffers:
            raise RuntimeError(f"Failed to load any valid noise audio files from {noise_dir}")

    def get_noise_chunk(self, target_len: int) -> torch.Tensor:
        """Constructs a noise tensor matching target_len by chaining random clips."""
        chunks = []
        cur_len = 0
        while cur_len < target_len:
            buf = random.choice(self.buffers)
            chunks.append(buf)
            cur_len += len(buf)
        noise = torch.cat(chunks, dim=0)[:target_len]
        return noise

    def mix(
        self,
        clean_waveform: torch.Tensor,
        snr_range: Tuple[float, float] = (0.0, 20.0),
    ) -> torch.Tensor:
        """
        Mixes clean waveform with real noise at a random SNR in dB:
        SNR = 10 * log10(P_clean / P_noise)
        clean_waveform: shape (1, T) or (T,)
        """
        is_2d = (clean_waveform.ndim == 2)
        w = clean_waveform.squeeze(0) if is_2d else clean_waveform

        target_len = w.shape[-1]
        noise = self.get_noise_chunk(target_len).to(w.device)

        clean_rms = torch.sqrt(torch.mean(w ** 2) + 1e-8)
        noise_rms = torch.sqrt(torch.mean(noise ** 2) + 1e-8)

        snr_db = random.uniform(snr_range[0], snr_range[1])
        snr_linear = 10.0 ** (snr_db / 20.0)

        scale = clean_rms / (noise_rms * snr_linear)
        noisy_w = torch.clamp(w + scale * noise, -1.0, 1.0)

        return noisy_w.unsqueeze(0) if is_2d else noisy_w
