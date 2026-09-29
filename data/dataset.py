import os
import csv
import random
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as T

from data.augment import SpecAugment, mixup_samples, RandomTimeShift, RandomNoise


class ACADDataset(Dataset):
    """
    Dataset for the Automatic Contextual Audio Denoising (ACAD) benchmark (Luong et al., EUSIPCO 2026).
    Yields paired (noisy_mel, clean_mel) log-Mel spectrograms, each of shape [1, 128, 500].

    Time resolution:
      hop_length = 160 at 16 kHz -> 100 frames / second (100 * time).
      duration_sec = 5.0 -> exactly 500 time frames.

    Expected directory hierarchy on disk:
      root_dir/
      └── {split}/ (e.g. train, val, test)
          └── {scene}/ (Kitchen, Park, Restaurant, Restroom, Street, Subway)
              └── {clip_id}/
                  ├── {clip_id}.wav                      <-- Noisy mixture
                  └── isolated_events/
                      └── background0_{scene}.wav        <-- Clean target scene
    """
    def __init__(
        self,
        root_dir="data/acad",
        split="train",
        sample_rate=16000,
        duration_sec=5.0,
        n_mels=128,
        n_fft=1024,
        win_length=400,
        hop_length=160,
        target_frames=500,
        scenes=None,
        mock=False,
        mock_length=256,
        use_augment=True,
        freq_mask_param=24,
        time_mask_param=48,
        num_freq_masks=2,
        num_time_masks=2,
        mixup_alpha=0.5,
        mixup_prob=0.5,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.sample_rate = sample_rate
        self.duration_sec = duration_sec
        self.target_len = int(sample_rate * duration_sec)
        self.target_frames = target_frames
        self.scenes = scenes
        self.mock = mock
        self.mock_length = mock_length
        self.use_augment = use_augment and (self.split == "train")
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            power=2.0,
        )

        self.spec_augment = SpecAugment(
            freq_mask_param=freq_mask_param,
            time_mask_param=time_mask_param,
            num_freq_masks=num_freq_masks,
            num_time_masks=num_time_masks,
        ) if self.use_augment else None

        if not self.mock:
            self.samples = self._index_samples()
        else:
            self.samples = []

    def _find_split_dir(self):
        direct_split = self.root_dir / self.split
        if direct_split.is_dir():
            return direct_split
        nested_split = self.root_dir / "acad" / self.split
        if nested_split.is_dir():
            return nested_split
        return direct_split

    def _index_samples(self):
        split_dir = self._find_split_dir()
        if not split_dir.is_dir():
            return []

        samples = []
        scene_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        if self.scenes:
            scene_dirs = [d for d in scene_dirs if d.name in self.scenes]

        for scene_dir in scene_dirs:
            for clip_dir in scene_dir.iterdir():
                if not clip_dir.is_dir():
                    continue

                # Locate noisy audio file ({clip_id}.wav)
                noisy_file = clip_dir / f"{clip_dir.name}.wav"
                if not noisy_file.is_file():
                    wavs = [f for f in clip_dir.glob("*.wav") if f.is_file()]
                    if wavs:
                        noisy_file = wavs[0]
                    else:
                        continue

                # Locate clean audio inside isolated_events/
                iso_dir = clip_dir / "isolated_events"
                if not iso_dir.is_dir():
                    continue

                clean_candidates = list(iso_dir.glob("background*.wav"))
                if not clean_candidates:
                    clean_candidates = list(iso_dir.glob("*.wav"))

                if clean_candidates:
                    clean_file = clean_candidates[0]
                    samples.append((str(noisy_file), str(clean_file)))

        return sorted(samples, key=lambda x: x[0])

    def _load_audio(self, file_path):
        waveform, sr = torchaudio.load(file_path)

        # Convert multi-channel to mono
        if waveform.ndim == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # Resample to target sample rate
        if sr != self.sample_rate:
            waveform = AF.resample(waveform, orig_freq=sr, new_freq=self.sample_rate)

        return waveform

    def _waveform_to_log_mel(self, waveform):
        # Compute Mel spectrogram: [1, n_mels, time_frames]
        mel = self.mel_transform(waveform)
        log_mel = torch.log(mel.clamp(min=1e-6))

        # Adjust time dimension to exact target_frames (500)
        if log_mel.shape[-1] > self.target_frames:
            log_mel = log_mel[..., :self.target_frames]
        elif log_mel.shape[-1] < self.target_frames:
            log_mel = F.pad(log_mel, (0, self.target_frames - log_mel.shape[-1]))

        return log_mel

    def __len__(self):
        if self.mock:
            return self.mock_length
        return len(self.samples)

    def _get_raw_item(self, idx):
        if self.mock:
            clean_mel = torch.randn(1, 128, self.target_frames)
            noisy_mel = clean_mel + 0.1 * torch.randn(1, 128, self.target_frames)
            return noisy_mel, clean_mel

        if not self.samples:
            split_path = self._find_split_dir()
            raise FileNotFoundError(
                f"No paired ACAD audio samples found in '{self.root_dir}'. "
                f"Expected directory structure: {split_path}/<Scene>/<ClipID>/<ClipID>.wav "
                "with isolated_events/background0_*.wav"
            )

        noisy_path, clean_path = self.samples[idx]
        noisy_wave = self._load_audio(noisy_path)
        clean_wave = self._load_audio(clean_path)

        # Align lengths between noisy and clean waveforms
        min_len = min(noisy_wave.shape[-1], clean_wave.shape[-1])
        noisy_wave = noisy_wave[..., :min_len]
        clean_wave = clean_wave[..., :min_len]

        # Synchronous cropping to 5.0 seconds (target_len = 80,000 samples)
        if min_len > self.target_len:
            if self.split == "train":
                max_start = min_len - self.target_len
                start = random.randint(0, max_start)
            else:
                start = (min_len - self.target_len) // 2
            noisy_wave = noisy_wave[..., start : start + self.target_len]
            clean_wave = clean_wave[..., start : start + self.target_len]
        elif min_len < self.target_len:
            pad = self.target_len - min_len
            noisy_wave = F.pad(noisy_wave, (0, pad))
            clean_wave = F.pad(clean_wave, (0, pad))

        noisy_mel = self._waveform_to_log_mel(noisy_wave)
        clean_mel = self._waveform_to_log_mel(clean_wave)

        return noisy_mel, clean_mel

    def __getitem__(self, idx):
        noisy_mel, clean_mel = self._get_raw_item(idx)

        if self.use_augment and self.spec_augment is not None:
            total_items = self.mock_length if self.mock else len(self.samples)
            if self.mixup_prob > 0.0 and random.random() < self.mixup_prob and total_items > 1:
                idx2 = random.randint(0, total_items - 2)
                if idx2 >= idx:
                    idx2 += 1
                noisy2, clean2 = self._get_raw_item(idx2)
                noisy_mel, clean_mel = mixup_samples(
                    noisy_mel, clean_mel, noisy2, clean2, alpha=self.mixup_alpha
                )

            noisy_mel = self.spec_augment(noisy_mel)

        return noisy_mel, clean_mel


class FSD50KDataset(Dataset):
    """
    Dataset for FSD50K multi-label sound event classification.
    Matches the Kaggle `yousirui1/fsd50k` dataset directory structure:

      root_dir/
      ├── FSD50K.ground_truth/
      │   ├── dev.csv              <-- fname, labels, mids, split ("train" / "val")
      │   ├── eval.csv             <-- fname, labels, mids ("eval" / "test")
      │   └── vocabulary.csv       <-- class index, label name, mid
      ├── FSD50K.dev_audio_16k/    <-- (or FSD50K.dev_audio / FSD50K.dev)
      └── FSD50K.eval_audio_16k/   <-- (or FSD50K.eval_audio / FSD50K.eval)
    """
    def __init__(
        self,
        root_dir="data/fsd50k",
        split="train",
        sample_rate=16000,
        duration_sec=10.0,
        n_mels=128,
        n_fft=1024,
        win_length=400,
        hop_length=160,
        target_frames=1000,
        mock=False,
        mock_length=256,
        num_classes=200,
        use_augment=True,
        freq_mask_param=48,
        time_mask_param=192,
        num_freq_masks=2,
        num_time_masks=2,
        mixup_alpha=0.5,
        mixup_prob=0.5,
        time_shift_param=10,
        noise_param=0.05,
        normalize=True,
        norm_mean=-4.2677393,
        norm_std=4.5689974,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split.lower()
        self.sample_rate = sample_rate
        self.duration_sec = duration_sec
        self.target_len = int(sample_rate * duration_sec)
        self.target_frames = target_frames
        self.mock = mock
        self.mock_length = mock_length
        self.num_classes = num_classes
        self.use_augment = use_augment and (self.split == "train")
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob
        self.normalize = normalize
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            power=2.0,
        )

        self.time_shift = RandomTimeShift(max_shift=time_shift_param) if (self.use_augment and time_shift_param > 0) else None
        self.noise = RandomNoise(max_noise=noise_param) if (self.use_augment and noise_param > 0) else None

        self.spec_augment = SpecAugment(
            freq_mask_param=freq_mask_param,
            time_mask_param=time_mask_param,
            num_freq_masks=num_freq_masks,
            num_time_masks=num_time_masks,
        ) if self.use_augment else None

        if not self.mock:
            self.label_to_idx, self.idx_to_label = self._load_vocabulary()
            self.samples = self._load_samples()
        else:
            self.label_to_idx = {f"class_{i}": i for i in range(num_classes)}
            self.idx_to_label = {i: f"class_{i}" for i in range(num_classes)}
            self.samples = []

    def _find_dir(self, candidate_names):
        for name in candidate_names:
            p = self.root_dir / name
            if p.is_dir():
                return p
        return None

    def _load_vocabulary(self):
        gt_dir = self._find_dir(["FSD50K.ground_truth", "ground_truth", "metadata"])
        vocab_file = None
        if gt_dir:
            v_cand = gt_dir / "vocabulary.csv"
            if v_cand.is_file():
                vocab_file = v_cand

        if vocab_file is None:
            v_cand = self.root_dir / "vocabulary.csv"
            if v_cand.is_file():
                vocab_file = v_cand

        label_to_idx = {}
        idx_to_label = {}

        if vocab_file and vocab_file.is_file():
            with open(vocab_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue
                    try:
                        idx = int(row[0].strip())
                        label = row[1].strip()
                    except ValueError:
                        try:
                            idx = int(row[1].strip())
                            label = row[0].strip()
                        except ValueError:
                            continue
                    label_to_idx[label] = idx
                    idx_to_label[idx] = label

        return label_to_idx, idx_to_label

    def _load_samples(self):
        gt_dir = self._find_dir(["FSD50K.ground_truth", "ground_truth", "metadata"])
        if gt_dir is None:
            gt_dir = self.root_dir

        is_eval_split = self.split in ["eval", "test"]
        csv_name = "eval.csv" if is_eval_split else "dev.csv"
        csv_path = gt_dir / csv_name
        if not csv_path.is_file():
            csv_path = self.root_dir / csv_name

        if not csv_path.is_file():
            return []

        if is_eval_split:
            audio_dir = self._find_dir([
                "FSD50K.eval_audio_16k",
                "FSD50K.eval_audio",
                "FSD50K.eval",
                "eval_audio",
                "eval"
            ])
        else:
            audio_dir = self._find_dir([
                "FSD50K.dev_audio_16k",
                "FSD50K.dev_audio",
                "FSD50K.dev",
                "dev_audio",
                "dev"
            ])

        samples = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not is_eval_split and "split" in row:
                    row_split = row["split"].strip().lower()
                    if row_split != self.split:
                        continue

                fname = row["fname"].strip()
                labels_str = row.get("labels", "").strip()
                label_list = [l.strip() for l in labels_str.split(",") if l.strip()]

                if audio_dir:
                    audio_path = audio_dir / f"{fname}.wav"
                else:
                    audio_path = self.root_dir / f"{fname}.wav"

                samples.append((str(audio_path), label_list))

        return samples

    def _load_and_crop_audio(self, file_path):
        waveform, sr = torchaudio.load(file_path)

        if waveform.ndim == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        if sr != self.sample_rate:
            waveform = AF.resample(waveform, orig_freq=sr, new_freq=self.sample_rate)

        length = waveform.shape[-1]
        if length > self.target_len:
            if self.split == "train":
                max_start = length - self.target_len
                start = random.randint(0, max_start)
            else:
                start = (length - self.target_len) // 2
            waveform = waveform[..., start : start + self.target_len]
        elif length < self.target_len:
            pad = self.target_len - length
            waveform = F.pad(waveform, (0, pad))

        mel = self.mel_transform(waveform)
        log_mel = torch.log(mel.clamp(min=1e-6))

        if self.normalize:
            log_mel = (log_mel - self.norm_mean) / (self.norm_std * 2.0)

        if log_mel.shape[-1] > self.target_frames:
            log_mel = log_mel[..., :self.target_frames]
        elif log_mel.shape[-1] < self.target_frames:
            log_mel = F.pad(log_mel, (0, self.target_frames - log_mel.shape[-1]))

        return log_mel

    def __len__(self):
        if self.mock:
            return self.mock_length
        return len(self.samples)

    def _get_raw_item(self, idx):
        if self.mock:
            mel = torch.randn(1, 128, self.target_frames)
            target = torch.zeros(self.num_classes, dtype=torch.float32)
            active_classes = torch.randint(0, self.num_classes, (random.randint(1, 3),))
            target[active_classes] = 1.0
            return mel, target

        if not self.samples:
            raise FileNotFoundError(
                f"No FSD50K samples found for split '{self.split}' in '{self.root_dir}'. "
                "Ensure ground truth CSVs and audio directories are extracted matching yousirui1/fsd50k."
            )

        audio_path, labels = self.samples[idx]
        mel = self._load_and_crop_audio(audio_path)

        target = torch.zeros(self.num_classes, dtype=torch.float32)
        for label in labels:
            if label in self.label_to_idx:
                target[self.label_to_idx[label]] = 1.0

        return mel, target

    def __getitem__(self, idx):
        mel, target = self._get_raw_item(idx)

        if self.use_augment:
            total_items = self.mock_length if self.mock else len(self.samples)
            if self.mixup_prob > 0.0 and random.random() < self.mixup_prob and total_items > 1:
                idx2 = random.randint(0, total_items - 2)
                if idx2 >= idx:
                    idx2 += 1
                mel2, target2 = self._get_raw_item(idx2)
                mel, target = mixup_samples(
                    mel, target, mel2, target2, alpha=self.mixup_alpha
                )

            if self.time_shift is not None:
                mel = self.time_shift(mel)

            if self.noise is not None:
                mel = self.noise(mel)

            if self.spec_augment is not None:
                mel = self.spec_augment(mel)

        return mel, target

    def get_sample_weights(self):
        """
        Computes inverse class-frequency sample weights for class-balanced sampling.
        Matches AST and PSLA class balancing on FSD50K.
        """
        if self.mock or not self.samples:
            return torch.ones(len(self), dtype=torch.double)

        class_counts = torch.zeros(self.num_classes, dtype=torch.double)
        sample_indices_list = []
        for _, labels in self.samples:
            idxs = [self.label_to_idx[l] for l in labels if l in self.label_to_idx]
            for idx in idxs:
                class_counts[idx] += 1.0
            sample_indices_list.append(idxs)

        class_weights = 1.0 / torch.clamp(class_counts, min=1.0)

        sample_weights = torch.zeros(len(self.samples), dtype=torch.double)
        for i, idxs in enumerate(sample_indices_list):
            if idxs:
                sample_weights[i] = class_weights[idxs].sum()
            else:
                sample_weights[i] = 1.0

        return sample_weights
