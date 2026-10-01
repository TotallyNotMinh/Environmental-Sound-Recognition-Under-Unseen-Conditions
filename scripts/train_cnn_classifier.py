import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import argparse
import csv
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from models import CNNClassifier, ARCH_CONFIGS
from data.dataset import FSD50KDataset
from data.sampler import DistributedWeightedSampler
from metrics.classification import MultiLabelClassificationMetrics

parser = argparse.ArgumentParser(description="Train a CMKD-style EfficientNet CNN classifier on FSD50K multi-label sound events")
parser.add_argument("--batch-size", type=int, default=24, help="EFFECTIVE total batch per optimizer step, split across GPUs and --grad-accum-steps (CMKD FSD50K CNN: 24)")
parser.add_argument("--duration-sec", type=float, default=10.0, help="Audio clip duration in seconds")
parser.add_argument("--target-frames", type=int, default=1000, help="Target spectrogram frames")
parser.add_argument("--checkpoint-path", type=str, default=None, help="Resume training checkpoint")
parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/fsd50k_cnn_b0/", help="Directory to save checkpoints")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps")
parser.add_argument("--data-path", type=str, default="data/fsd50k", help="Path to FSD50K dataset root")
parser.add_argument("--num-epoch", type=int, default=50, help="Number of training epochs (CMKD FSD50K default: 50)")
parser.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate (CMKD FSD50K CNN default: 5e-4)")
parser.add_argument("--weight-decay", type=float, default=5e-7, help="Adam weight decay (PSLA/AST released code: 5e-7)")
parser.add_argument("--adam-beta1", type=float, default=0.95, help="Adam beta1 (PSLA/AST released code: 0.95; beta2 stays 0.999)")
parser.add_argument("--warmup-steps", type=int, default=1000, help="Linear LR warmup over the first N optimizer steps, updated every 50 steps (PSLA/AST released code: 1000). 0 disables")
parser.add_argument("--grad-clip", type=float, default=0.0, help="Max grad norm; 0 disables (PSLA/AST released code: no clipping)")
parser.add_argument("--norm-mean", type=float, default=-4.6476, help="Spectrogram normalization mean (PSLA FSD50K recipe: -4.6476)")
parser.add_argument("--norm-std", type=float, default=4.5699, help="Spectrogram normalization std (PSLA FSD50K recipe: 4.5699)")
parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers per GPU")
parser.add_argument("--patience", type=int, default=0, help="Early stopping patience (epochs without mAP improvement); 0 disables -- CMKD trains the full epoch count")
parser.add_argument("--val-interval", type=int, default=1, help="Validation frequency (in epochs)")
parser.add_argument("--mock", action="store_true", help="Use synthetic mock data for testing/benchmarking")
parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation (SpecAugment, Mixup, TimeShift, Noise)")
parser.add_argument("--freq-mask", type=int, default=48, help="SpecAugment frequency mask parameter (CMKD FSD50K default: 48)")
parser.add_argument("--time-mask", type=int, default=192, help="SpecAugment time mask parameter (CMKD FSD50K default: 192)")
parser.add_argument("--time-shift", type=int, default=10, help="Random time shift in frames (CMKD default: 10)")
parser.add_argument("--noise-level", type=float, default=0.05, help="Spectrogram uniform noise level (CMKD default: 0.05)")
parser.add_argument("--label-smoothing", type=float, default=0.1, help="BCE label smoothing factor (CMKD default: 0.1)")
parser.add_argument("--no-class-balancing", action="store_true", help="Disable class-balanced sampling")
parser.add_argument("--mixup-alpha", type=float, default=0.5, help="Mixup beta distribution alpha parameter")
parser.add_argument("--mixup-prob", type=float, default=0.5, help="Probability of applying Mixup per sample")
parser.add_argument("--arch", type=str, default="b0", choices=list(ARCH_CONFIGS.keys()), help="EfficientNet backbone (default: b0, CMKD's optimal teacher for AST-Base)")
parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet-pretrained backbone initialization")
parser.add_argument("--lr-patience", type=int, default=2, help="Epochs with no val mAP improvement before LR is halved (CMKD default: 2)")
parser.add_argument("--lr-factor", type=float, default=0.5, help="LR reduction factor on plateau (CMKD default: 0.5)")

args = parser.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def append_metrics_csv(checkpoint_dir, row):
    """Append one epoch's metrics to <checkpoint_dir>/metrics.csv (header written once; survives resume)."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    csv_path = os.path.join(checkpoint_dir, "metrics.csv")
    write_header = not os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(checkpoint_dir, checkpoint_name, epoch, model, optimizer, scheduler, best_map, epochs_without_improvement, scaler):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_save_path = os.path.join(checkpoint_dir, checkpoint_name)

    raw_model = model.module if hasattr(model, "module") else model
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_map": best_map,
        "epochs_without_improvement": epochs_without_improvement,
    }
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    torch.save(checkpoint, checkpoint_save_path)


def load_resume_checkpoint(checkpoint_path, device, model, optimizer, scheduler, scaler):
    if checkpoint_path is None or not os.path.isfile(checkpoint_path):
        return 1, 0.0, 0

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(state_dict, strict=True)

    if "optimizer_state_dict" in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    best_map = checkpoint.get("best_map", 0.0)
    epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)

    print(f"Resumed CNN checkpoint from {checkpoint_path}: start epoch {start_epoch}, best mAP: {best_map:.4f}")
    return start_epoch, best_map, epochs_without_improvement


def train():
    # ============== Distributed Setup ==============
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main = (rank == 0)
    set_seed(args.seed + rank)

    EPOCHS = args.num_epoch
    # --batch-size is the EFFECTIVE total batch per optimizer step:
    # per-GPU micro-batch = total / (num GPUs * grad-accum steps).
    # (EfficientNet has BatchNorm, so accumulation is not exactly equivalent to a
    # bigger batch here -- prefer --grad-accum-steps 1 for the CNN if memory allows.)
    micro_divisor = world_size * args.grad_accum_steps
    if args.batch_size % micro_divisor != 0:
        raise ValueError(
            f"--batch-size {args.batch_size} (total) must be divisible by "
            f"world size {world_size} x --grad-accum-steps {args.grad_accum_steps}"
        )
    BATCH_SIZE = args.batch_size // micro_divisor  # per-GPU micro-batch
    NUM_CLASSES = 200
    patience = args.patience
    checkpoint_path = args.checkpoint_path
    checkpoint_dir = args.checkpoint_dir

    # ============== Loss Function ==============
    criterion = nn.BCEWithLogitsLoss().to(device)

    # ============== Datasets & Loaders ==============
    train_dataset = FSD50KDataset(
        root_dir=args.data_path,
        split="train",
        duration_sec=args.duration_sec,
        target_frames=args.target_frames,
        mock=args.mock,
        num_classes=NUM_CLASSES,
        use_augment=not args.no_augment,
        freq_mask_param=args.freq_mask,
        time_mask_param=args.time_mask,
        time_shift_param=args.time_shift,
        noise_param=args.noise_level,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
        normalize=True,
        norm_mean=args.norm_mean,
        norm_std=args.norm_std,
    )
    val_dataset = FSD50KDataset(
        root_dir=args.data_path,
        split="val",
        duration_sec=args.duration_sec,
        target_frames=args.target_frames,
        mock=args.mock,
        num_classes=NUM_CLASSES,
        normalize=True,
        norm_mean=args.norm_mean,
        norm_std=args.norm_std,
    )

    if not args.no_class_balancing and not args.mock:
        sample_weights = train_dataset.get_sample_weights()
        if is_distributed:
            train_sampler = DistributedWeightedSampler(
                dataset=train_dataset,
                weights=sample_weights,
                num_replicas=world_size,
                rank=rank,
                replacement=True,
                seed=args.seed,
            )
        else:
            train_sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
    elif is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=seed_worker,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
    ) if is_main else None

    # ============== Model Initialization ==============
    arch_cfg = ARCH_CONFIGS[args.arch]
    model = CNNClassifier(
        arch=args.arch,
        num_classes=NUM_CLASSES,
        pretrained=not args.no_pretrained,
        dropout=0.0,
    ).to(device)

    # ============== Optimizer & Scheduler ==============
    # CMKD's CNN recipe uses a single Adam param group (unlike AST's split
    # encoder/head LR) and a validation-plateau schedule, not a fixed step decay.
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(args.adam_beta1, 0.999)
    )

    # mode="max" because mAP is "higher is better" -- the default mode="min"
    # would halve the LR exactly when it shouldn't.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=args.lr_factor, patience=args.lr_patience
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Resume checkpoint if specified
    start_epoch, best_map, epochs_without_improvement = load_resume_checkpoint(
        checkpoint_path, device, model, optimizer, scheduler, scaler
    )

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    metrics = MultiLabelClassificationMetrics(num_classes=NUM_CLASSES)

    if is_main:
        print(f"[Rank 0] FSD50K CNN Classifier Training (CMKD CNN-teacher recipe):")
        print(f"  • Device:                 {device} (world size: {world_size})")
        print(f"  • Total samples:          {len(train_dataset)} train, {len(val_dataset)} val")
        print(f"  • Effective batch size:   {BATCH_SIZE * world_size * args.grad_accum_steps}")
        print(f"  • Architecture:           {arch_cfg['name']}")
        print(f"  • Pretrained Backbone:    {'ImageNet (torchvision)' if not args.no_pretrained else 'Random init'}")
        sampler_desc = (
            "Class-Balanced (DistributedWeightedSampler)"
            if (not args.no_class_balancing and not args.mock and is_distributed)
            else (
                "Class-Balanced (WeightedRandomSampler)"
                if (not args.no_class_balancing and not args.mock)
                else (
                    "Standard DistributedSampler"
                    if is_distributed
                    else "Standard Uniform"
                )
            )
        )
        print(f"  • Sampling:               {sampler_desc}")

    # ============== Training and Validation Loop ==============
    # Warmup only on a fresh run; a resumed run is already past it.
    global_step = 0
    warmup_enabled = args.warmup_steps > 0 and start_epoch == 1

    for epoch in range(start_epoch, EPOCHS + 1):
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        model.train(True)
        running_train_loss = 0.0
        accum_steps = args.grad_accum_steps
        optimizer.zero_grad(set_to_none=True)

        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch [{epoch:02d}/{EPOCHS:02d}] (Train)",
            leave=False,
            disable=not is_main,
        )
        for batch_idx, (spectrograms, targets) in enumerate(train_pbar):
            spectrograms = spectrograms.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(spectrograms)
                if args.label_smoothing > 0.0:
                    smoothed_targets = targets * (1.0 - args.label_smoothing) + 0.5 * args.label_smoothing
                    loss = criterion(logits, smoothed_targets)
                else:
                    loss = criterion(logits, targets)
                loss_to_backward = loss / accum_steps

            scaler.scale(loss_to_backward).backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                # PSLA/AST warmup: every 50 steps within the first warmup_steps,
                # set lr = (step / warmup_steps) * base lr (step 0 -> lr 0, as in their code).
                if warmup_enabled and global_step <= args.warmup_steps and global_step % 50 == 0:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = (global_step / args.warmup_steps) * args.lr
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            running_train_loss += loss.item()
            if is_main:
                train_pbar.set_postfix({"bce": f"{loss.item():.4f}"})

        avg_train_loss = running_train_loss / max(1, len(train_loader))

        if is_distributed:
            loss_tensor = torch.tensor([avg_train_loss], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_train_loss = (loss_tensor / world_size).item()

        # ============== Validation Loop (Rank 0) ==============
        val_map_for_scheduler = None
        val_results = None
        avg_val_loss = None
        if is_main and (epoch % args.val_interval == 0 or epoch == EPOCHS):
            running_val_loss = 0.0
            metrics.reset()

            raw_model = model.module if hasattr(model, "module") else model
            raw_model.eval()

            val_pbar = tqdm(
                val_loader,
                desc=f"Epoch [{epoch:02d}/{EPOCHS:02d}] (Val)  ",
                leave=False,
            )
            with torch.no_grad():
                for spectrograms, targets in val_pbar:
                    spectrograms = spectrograms.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)

                    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                        logits = raw_model(spectrograms)
                        val_loss = criterion(logits, targets)

                    running_val_loss += val_loss.item()
                    metrics.update(logits, targets)
                    val_pbar.set_postfix({"val_bce": f"{val_loss.item():.4f}"})

            avg_val_loss = running_val_loss / max(1, len(val_loader))
            val_results = metrics.compute()
            val_map_for_scheduler = val_results["mAP"]

            del val_pbar
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ============== Sync the plateau scheduler across ranks ==============
        # Only rank 0 computed val_map above; every rank must step the
        # scheduler with the *same* value so their optimizer LRs stay in
        # sync (mirrors the should_stop/avg_train_loss broadcast pattern
        # already used elsewhere in this repo's training scripts).
        if is_distributed:
            val_map_tensor = torch.tensor(
                [val_map_for_scheduler if val_map_for_scheduler is not None else -1.0],
                device=device,
            )
            dist.broadcast(val_map_tensor, src=0)
            broadcast_value = val_map_tensor.item()
            val_map_for_scheduler = None if broadcast_value < 0 else broadcast_value

        if val_map_for_scheduler is not None:
            scheduler.step(val_map_for_scheduler)

        # ============== Rank-0 bookkeeping: print + checkpoint ==============
        # Runs AFTER the scheduler step (and after best_map/epochs_without_improvement
        # are updated for THIS epoch), so checkpoint.pth's scheduler_state_dict and
        # best-tracking fields are never one epoch stale on resume.
        if is_main and val_results is not None:
            val_map = val_results["mAP"]
            val_mauc = val_results["mAUC"]
            micro_f1 = val_results["micro_f1"]
            macro_f1 = val_results["macro_f1"]
            top1_hit = val_results["top1_hit"]
            top5_hit = val_results["top5_hit"]
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"=== Epoch [{epoch:02d}/{EPOCHS:02d}] | "
                f"Train BCE: {avg_train_loss:.4f} | "
                f"Val BCE: {avg_val_loss:.4f} | "
                f"mAP: {val_map:.4f} | "
                f"mAUC: {val_mauc:.4f} | "
                f"Micro-F1: {micro_f1:.4f} | "
                f"Macro-F1: {macro_f1:.4f} | "
                f"Top-1 Hit: {top1_hit:.4f} | "
                f"Top-5 Hit: {top5_hit:.4f} | "
                f"LR: {current_lr:.6f} ==="
            )
            append_metrics_csv(checkpoint_dir, {
                "epoch": epoch,
                "train_bce": f"{avg_train_loss:.6f}",
                "val_bce": f"{avg_val_loss:.6f}",
                "mAP": f"{val_map:.6f}",
                "mAUC": f"{val_mauc:.6f}",
                "micro_f1": f"{micro_f1:.6f}",
                "macro_f1": f"{macro_f1:.6f}",
                "top1_hit": f"{top1_hit:.6f}",
                "top5_hit": f"{top5_hit:.6f}",
                "lr": f"{current_lr:.8g}",
            })

            if val_map > best_map:
                best_map = val_map
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            save_checkpoint(
                checkpoint_dir, "checkpoint.pth", epoch, model, optimizer, scheduler,
                best_map, epochs_without_improvement, scaler,
            )

            if epochs_without_improvement == 0:
                save_checkpoint(
                    checkpoint_dir, "best_cnn.pth", epoch, model, optimizer, scheduler,
                    best_map, epochs_without_improvement, scaler,
                )
                print(f"  --> Saved new best CNN checkpoint (Val mAP: {best_map:.4f})")

            del val_results

        # Synchronize early stopping decision across ranks
        should_stop = torch.tensor(
            [1.0 if (patience > 0 and epochs_without_improvement >= patience) else 0.0],
            device=device,
        )
        if is_distributed:
            dist.broadcast(should_stop, src=0)
        if should_stop.item() == 1.0:
            if is_main:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

        if is_distributed:
            dist.barrier()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    train()
