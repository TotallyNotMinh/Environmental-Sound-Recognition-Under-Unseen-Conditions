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

# Direct import as requested: user will implement/customize Classifer in models
from models import Classifer
from data.dataset import FSD50KDataset
from data.sampler import DistributedWeightedSampler
from metrics.classification import MultiLabelClassificationMetrics

parser = argparse.ArgumentParser(description="Train AST Classifier on FSD50K multi-label sound events")
parser.add_argument("--batch-size", type=int, default=12, help="Batch size per GPU (AST default: 12)")
parser.add_argument("--duration-sec", type=float, default=10.0, help="Audio clip duration in seconds (AST default: 10.0)")
parser.add_argument("--target-frames", type=int, default=1000, help="Target spectrogram frames (AST default: 1000)")
parser.add_argument("--checkpoint-path", type=str, default=None, help="Resume training checkpoint")
parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/fsd50k/", help="Directory to save checkpoints")
parser.add_argument("--pretrained-encoder", type=str, default=None, help="Path to pretrained denoiser/encoder weights")
parser.add_argument("--freeze-encoder", action="store_true", help="Freeze encoder backbone for linear evaluation")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps")
parser.add_argument("--data-path", type=str, default="data/fsd50k", help="Path to FSD50K dataset root")
parser.add_argument("--num-epoch", type=int, default=30, help="Number of training epochs (FSD50K default: 50)")
parser.add_argument("--encoder-lr", type=float, default=5e-5, help="Backbone encoder learning rate (AST default: 5e-5)")
parser.add_argument("--head-lr", type=float, default=5e-5, help="Classifier head learning rate (AST default: 5e-5)")
parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers per GPU")
parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
parser.add_argument("--val-interval", type=int, default=1, help="Validation frequency (in epochs)")
parser.add_argument("--mock", action="store_true", help="Use synthetic mock data for testing/benchmarking")
parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation (SpecAugment, Mixup, TimeShift, Noise)")
parser.add_argument("--freq-mask", type=int, default=48, help="SpecAugment frequency mask parameter (AST default: 48)")
parser.add_argument("--time-mask", type=int, default=192, help="SpecAugment time mask parameter (AST default: 192)")
parser.add_argument("--time-shift", type=int, default=10, help="Random time shift in frames (CMKD default: 10)")
parser.add_argument("--noise-level", type=float, default=0.05, help="Spectrogram uniform noise level (CMKD default: 0.05)")
parser.add_argument("--label-smoothing", type=float, default=0.1, help="BCE label smoothing factor (CMKD default: 0.1)")
parser.add_argument("--no-class-balancing", action="store_true", help="Disable class-balanced sampling")
parser.add_argument("--mixup-alpha", type=float, default=0.5, help="Mixup beta distribution alpha parameter")
parser.add_argument("--mixup-prob", type=float, default=0.5, help="Probability of applying Mixup per sample")
parser.add_argument("--encoder", type=str, default="ast", choices=["ast", "resnet18", "resnet34", "resnet152"], help="Backbone architecture (default: ast)")
parser.add_argument("--arch", type=str, default="tiny", choices=["tiny", "small", "base"], help="ViT backbone architecture (default: tiny)")
parser.add_argument("--no-dino", action="store_true", help="Disable pretrained ViT backbone initialization")
parser.add_argument("--no-cls-dist", action="store_true", help="Disable CLS+DIST dual token pooling (fall back to mean pooling)")
parser.add_argument("--lr-scheduler", type=str, default="ast_step", choices=["ast_step", "cosine"], help="LR scheduler: ast_step (decay 0.90 after epoch 5) or cosine")

ARCH_CONFIGS = {
    "tiny": {"tok_dim": 192, "num_head": 3, "num_layer": 12, "name": "DeiT ViT-Tiny"},
    "small": {"tok_dim": 384, "num_head": 6, "num_layer": 12, "name": "DeiT ViT-Small"},
    "base": {"tok_dim": 768, "num_head": 12, "num_layer": 12, "name": "DeiT ViT-Base"},
}

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


def load_pretrained_encoder_weights(model, pretrained_path, device, arch_name="ViT", is_main=True):
    """
    Extracts and loads pretrained ASTEncoder weights from a Denoiser pretraining checkpoint
    or an earlier AST checkpoint, cleanly discarding unused decoder/head parameters.
    """
    if pretrained_path is None or not os.path.isfile(pretrained_path):
        if is_main:
            raw_model = model.module if hasattr(model, "module") else model
            has_dino = getattr(raw_model.encoder, "use_dino", False) if hasattr(raw_model, "encoder") else False
            if has_dino:
                print(f"No custom checkpoint specified; using pretrained {arch_name} encoder.")
            else:
                print("No pretrained encoder specified. Training classifier from random initialization.")
        return

    checkpoint = torch.load(pretrained_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    # Clean DDP module. prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    raw_model = model.module if hasattr(model, "module") else model
    encoder_module = raw_model.encoder if hasattr(raw_model, "encoder") else raw_model

    encoder_state = encoder_module.state_dict()
    extracted_weights = {}

    for k, v in state_dict.items():
        if k.startswith("encoder."):
            clean_k = k[len("encoder."):]
            if clean_k in encoder_state and encoder_state[clean_k].shape == v.shape:
                extracted_weights[clean_k] = v
        elif k in encoder_state and encoder_state[k].shape == v.shape:
            extracted_weights[k] = v

    missing, unexpected = encoder_module.load_state_dict(extracted_weights, strict=False)
    if is_main:
        print(f"Successfully warm-started ASTEncoder from: {pretrained_path}")
        print(f"  • Loaded {len(extracted_weights)}/{len(encoder_state)} encoder layers.")
        if missing:
            print(f"  • Missing layers (kept initialized): {len(missing)}")


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

    print(f"Resumed classification checkpoint from {checkpoint_path}: start epoch {start_epoch}, best mAP: {best_map:.4f}")
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
    BATCH_SIZE = args.batch_size
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
    )
    val_dataset = FSD50KDataset(
        root_dir=args.data_path,
        split="val",
        duration_sec=args.duration_sec,
        target_frames=args.target_frames,
        mock=args.mock,
        num_classes=NUM_CLASSES,
        normalize=True,
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
    model = Classifer(
        encoder_type=args.encoder,
        tok_dim=arch_cfg["tok_dim"],
        num_classes=NUM_CLASSES,
        c_in=1,
        overlap=6,
        patch_size=16,
        size=(128, args.target_frames),
        num_head=arch_cfg["num_head"],
        num_layer=arch_cfg["num_layer"],
        pretrained_dino=(not args.no_dino and args.pretrained_encoder is None),
        use_cls_dist=not args.no_cls_dist,
    ).to(device)

    # Load pretrained encoder weights if supplied
    load_pretrained_encoder_weights(model, args.pretrained_encoder, device, arch_name=arch_cfg["name"], is_main=is_main)

    # Freeze encoder parameters if linear probe requested
    raw_model = model.module if hasattr(model, "module") else model
    if args.freeze_encoder and hasattr(raw_model, "encoder"):
        for param in raw_model.encoder.parameters():
            param.requires_grad = False
        if is_main:
            print("Frozen ASTEncoder backbone for linear probing evaluation.")

    # ============== Optimizer & Schedulers ==============
    if hasattr(raw_model, "encoder") and not args.freeze_encoder:
        encoder_params = [p for p in raw_model.encoder.parameters() if p.requires_grad]
        head_params = [p for n, p in raw_model.named_parameters() if not n.startswith("encoder") and p.requires_grad]
        param_groups = [
            {"params": encoder_params, "lr": args.encoder_lr, "weight_decay": args.weight_decay},
            {"params": head_params, "lr": args.head_lr, "weight_decay": args.weight_decay},
        ]
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        param_groups = [{"params": trainable_params, "lr": args.head_lr, "weight_decay": args.weight_decay}]

    optimizer = torch.optim.AdamW(param_groups)

    if args.lr_scheduler == "ast_step":
        # AST schedule: keep initial LR for 5 epochs, then decay by 0.90 every epoch
        def ast_lr_lambda(ep):
            if ep < 5:
                return 1.0
            return 0.90 ** (ep - 4)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=ast_lr_lambda)
    else:
        warmup_epochs = min(5, max(1, EPOCHS // 10))
        linear_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, EPOCHS - warmup_epochs), eta_min=1e-6
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[linear_scheduler, cosine_scheduler], milestones=[warmup_epochs]
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
        print(f"[Rank 0] FSD50K Multi-Label Classification Training:")
        print(f"  • Device:                 {device} (world size: {world_size})")
        print(f"  • Total samples:          {len(train_dataset)} train, {len(val_dataset)} val")
        print(f"  • Effective batch size:   {BATCH_SIZE * world_size * args.grad_accum_steps}")
        backbone_desc = args.pretrained_encoder if args.pretrained_encoder else (f"{arch_cfg['name']} (pretrained)" if not args.no_dino else "Random init")
        print(f"  • Architecture:           ViT-{args.arch.capitalize()} ({arch_cfg['name']})")
        print(f"  • Pretrained Backbone:    {backbone_desc}")
        print(f"  • Encoder Frozen:         {args.freeze_encoder}")
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
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_train_loss += loss.item()
            if is_main:
                train_pbar.set_postfix({"bce": f"{loss.item():.4f}"})

        scheduler.step()
        avg_train_loss = running_train_loss / max(1, len(train_loader))

        if is_distributed:
            loss_tensor = torch.tensor([avg_train_loss], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_train_loss = (loss_tensor / world_size).item()

        # ============== Validation Loop (Rank 0) ==============
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

            val_map = val_results["mAP"]
            val_mauc = val_results["mAUC"]
            micro_f1 = val_results["micro_f1"]
            macro_f1 = val_results["macro_f1"]
            top1_hit = val_results["top1_hit"]
            top5_hit = val_results["top5_hit"]

            current_lr = scheduler.get_last_lr()[0]
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

            # Save latest checkpoint
            save_checkpoint(
                checkpoint_dir,
                "checkpoint.pth",
                epoch,
                model,
                optimizer,
                scheduler,
                best_map,
                epochs_without_improvement,
                scaler,
            )

            # Check for new best mAP
            if val_map > best_map:
                best_map = val_map
                epochs_without_improvement = 0
                save_checkpoint(
                    checkpoint_dir,
                    "best_classifier.pth",
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    best_map,
                    epochs_without_improvement,
                    scaler,
                )
                print(f"  --> Saved new best classifier checkpoint (Val mAP: {best_map:.4f})")
            else:
                epochs_without_improvement += 1

            del val_pbar, val_results
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
