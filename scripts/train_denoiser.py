import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import argparse
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from models.denoiser import Denoiser
from data.dataset import FSD50KDataset

parser = argparse.ArgumentParser(description="Pretrain Audio Denoiser on FSD50K dataset")
parser.add_argument("--batch-size", type=int, default=16, help="Batch size per GPU")
parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to resume training checkpoint")
parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/", help="Directory to save checkpoints")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps")
parser.add_argument("--data-path", type=str, default="data/fsd50k", help="Path to FSD50K dataset directory")
parser.add_argument("--noise-path", type=str, default=None, help="Path to real-world noise directory (e.g. TAU Urban Acoustic Scenes)")
parser.add_argument("--min-snr", type=float, default=0.0, help="Minimum SNR in dB for noise mixing")
parser.add_argument("--max-snr", type=float, default=20.0, help="Maximum SNR in dB for noise mixing")
parser.add_argument("--num-epoch", type=int, default=100, help="Number of training epochs")
parser.add_argument("--encoder-lr", type=float, default=1e-4, help="Encoder learning rate")
parser.add_argument("--decoder-lr", type=float, default=2e-4, help="Decoder learning rate")
parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers per GPU (<= 8)")
parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
parser.add_argument("--val-interval", type=int, default=1, help="Validation frequency (in epochs)")
parser.add_argument("--mock", action="store_true", help="Use synthetic mock data for testing/benchmarking")
parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation (SpecAugment and Mixup)")
parser.add_argument("--freq-mask", type=int, default=24, help="SpecAugment frequency mask parameter")
parser.add_argument("--time-mask", type=int, default=48, help="SpecAugment time mask parameter")
parser.add_argument("--mixup-alpha", type=float, default=0.5, help="Mixup alpha parameter")
parser.add_argument("--mixup-prob", type=float, default=0.5, help="Probability of applying Mixup per sample")

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


def save_checkpoint(checkpoint_dir, checkpoint_name, epoch, model, optimizer, scheduler, best_val_loss, epochs_without_improvement, scaler):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_save_path = os.path.join(checkpoint_dir, checkpoint_name)
    
    raw_model = model.module if hasattr(model, "module") else model
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "epochs_without_improvement": epochs_without_improvement,
    }
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    torch.save(checkpoint, checkpoint_save_path)


def load_checkpoint(checkpoint_path, device, model, optimizer, scheduler, scaler):
    if checkpoint_path is None or not os.path.isfile(checkpoint_path):
        print("No checkpoint detected. Starting training from scratch.")
        return 1, float("inf"), 0

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
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)

    print(f"Resumed checkpoint from {checkpoint_path}: start epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}")
    return start_epoch, best_val_loss, epochs_without_improvement


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

    # ============== Hyperparameters ==============
    EPOCHS = args.num_epoch
    BATCH_SIZE = args.batch_size
    patience = args.patience
    checkpoint_path = args.checkpoint_path
    checkpoint_dir = args.checkpoint_dir

    # ============== Losses ==============
    crit_l1 = nn.L1Loss().to(device)
    crit_mse = nn.MSELoss().to(device)

    # ============== Datasets & Distributed Samplers ==============
    train_dataset = FSD50KDataset(
        root_dir=args.data_path,
        split="train",
        mock=args.mock,
        use_augment=not args.no_augment,
        freq_mask_param=args.freq_mask,
        time_mask_param=args.time_mask,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
        return_labels=False,
        noise_dir=args.noise_path,
        min_snr=args.min_snr,
        max_snr=args.max_snr,
    )
    val_dataset = FSD50KDataset(
        root_dir=args.data_path,
        split="val",
        mock=args.mock,
        return_labels=False,
        noise_dir=args.noise_path,
        min_snr=args.min_snr,
        max_snr=args.max_snr,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False) if is_distributed else None

    num_workers = min(args.num_workers, 8)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker,
    ) if is_main else None

    # ============== Model & Architecture ==============
    # AST Spectrogram input: (B, 1, 128, 1000)
    model = Denoiser().to(device)
    raw_model = model.module if hasattr(model, "module") else model

     # ============== Optimizer & Schedulers ==============
    encoder_params = [p for p in raw_model.encoder.parameters() if p.requires_grad]
    decoder_params = [p for n, p in raw_model.named_parameters() if not n.startswith("encoder") and p.requires_grad]
    param_groups = [
        {"params": encoder_params, "lr": args.encoder_lr, "weight_decay": args.weight_decay},
        {"params": decoder_params, "lr": args.decoder_lr, "weight_decay": args.weight_decay},
    ]

    optimizer = torch.optim.AdamW(param_groups)

    # AST schedule: keep initial LR for 5 epochs, then decay by 0.90 every epoch
    def ast_lr_lambda(ep):
        if ep < 5:
            return 1.0
        return 0.90 ** (ep - 4)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=ast_lr_lambda)

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Load checkpoint before DDP wrapping
    start_epoch, best_val_loss, epochs_without_improvement = load_checkpoint(
        checkpoint_path, device, model, optimizer, scheduler, scaler
    )

    # Wrap model with DistributedDataParallel
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False

    if is_main:
        print(f"[Rank 0] EfficientNet-B0 Audio Denoiser Pretraining :")
        print(f"  • Device:                 {device} (world size: {world_size})")
        print(f"  • Total samples:          {len(train_dataset)} train, {len(val_dataset)} val")
        print(f"  • Effective batch size:   {BATCH_SIZE * world_size * args.grad_accum_steps}")
        print(f"  • Architecture:           EfficientNet-B0 + DenoisingDecoder")
        print(f"  • Pretrained Backbone:    ImageNet Pretrained")
        print(f"  • Learning rates:         Encoder={args.encoder_lr}, Decoder={args.decoder_lr}")
        if args.noise_path:
            print(f"  • Real Noise Source:      {args.noise_path} (SNR: [{args.min_snr}, {args.max_snr}] dB)")

    # ============== Training and Validation Loop ==============
    for epoch in range(start_epoch, EPOCHS + 1):
        if is_distributed and train_sampler is not None:
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
        for batch_idx, (noisy, clean) in enumerate(train_pbar):
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred_clean = model(noisy)
                loss = crit_l1(pred_clean, clean)
                loss_to_backward = loss / accum_steps

            scaler.scale(loss_to_backward).backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_train_loss += loss.item()
            if is_main:
                train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_train_loss = running_train_loss / max(1, len(train_loader))

        if is_distributed:
            loss_tensor = torch.tensor([avg_train_loss], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_train_loss = (loss_tensor / world_size).item()

        # ============== Validation Loop (Rank 0) ==============
        if is_main and (epoch % args.val_interval == 0 or epoch == EPOCHS):
            running_val_l1 = 0.0
            running_val_mse = 0.0
            running_noisy_l1 = 0.0

            raw_model = model.module if hasattr(model, "module") else model
            raw_model.eval()

            val_pbar = tqdm(
                val_loader,
                desc=f"Epoch [{epoch:02d}/{EPOCHS:02d}] (Val)  ",
                leave=False,
            )
            with torch.no_grad():
                for noisy, clean in val_pbar:
                    noisy = noisy.to(device, non_blocking=True)
                    clean = clean.to(device, non_blocking=True)

                    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                        pred_clean = raw_model(noisy)
                        val_l1 = crit_l1(pred_clean, clean)
                        val_mse = crit_mse(pred_clean, clean)
                        noisy_l1 = crit_l1(noisy, clean)

                    running_val_l1 += val_l1.item()
                    running_val_mse += val_mse.item()
                    running_noisy_l1 += noisy_l1.item()
                    val_pbar.set_postfix({"val_mae": f"{val_l1.item():.4f}"})

            num_val_batches = max(1, len(val_loader))
            avg_val_l1 = running_val_l1 / num_val_batches
            avg_val_mse = running_val_mse / num_val_batches
            avg_noisy_l1 = running_noisy_l1 / num_val_batches
            mae_improvement = avg_noisy_l1 - avg_val_l1

            current_lr = scheduler.get_last_lr()[0]
            print(
                f"=== Epoch [{epoch:02d}/{EPOCHS:02d}] | "
                f"Train L1: {avg_train_loss:.4f} | "
                f"Val MAE: {avg_val_l1:.4f} | "
                f"Val MSE: {avg_val_mse:.4f} | "
                f"Raw Noisy MAE: {avg_noisy_l1:.4f} | "
                f"MAE Gain: {mae_improvement:+.4f} | "
                f"LR: {current_lr:.6f} ==="
            )

            # Save latest checkpoint
            save_checkpoint(
                checkpoint_dir,
                "checkpoint.pth",
                epoch,
                model,
                optimizer,
                scheduler,
                best_val_loss,
                epochs_without_improvement,
                scaler,
            )

            # Check for best reconstruction MAE
            if avg_val_l1 < best_val_loss:
                best_val_loss = avg_val_l1
                epochs_without_improvement = 0
                save_checkpoint(
                    checkpoint_dir,
                    "best_denoiser.pth",
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    best_val_loss,
                    epochs_without_improvement,
                    scaler,
                )
                print(f"  --> Saved new best denoiser checkpoint (Val MAE: {best_val_loss:.4f})")
            else:
                epochs_without_improvement += 1

            del val_pbar
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Synchronize early stopping decision across ranks
        should_stop = torch.tensor(
            [1.0 if epochs_without_improvement >= patience else 0.0],
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