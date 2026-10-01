import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import json
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import Classifer, CNNClassifier
from models import ARCH_CONFIGS as CNN_ARCH_CONFIGS
from data.dataset import FSD50KDataset
from metrics.classification import MultiLabelClassificationMetrics

parser = argparse.ArgumentParser(description="Evaluate a trained CNN or AST checkpoint on an FSD50K split (default: eval, the split CMKD reports)")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to evaluate (best_cnn.pth / best_kd_ast.pth / checkpoint.pth). Required unless --mock")
parser.add_argument("--data-path", type=str, default="data/fsd50k", help="FSD50K root (folder containing FSD50K.ground_truth/)")
parser.add_argument("--split", type=str, default="eval", choices=["eval", "val"], help="eval = FSD50K eval set (paper's reported numbers); val = the validation split used during training")
parser.add_argument("--model", type=str, default="auto", choices=["auto", "cnn", "ast"], help="Model family; auto-detected from the checkpoint keys")
parser.add_argument("--no-cls-dist", action="store_true", help="AST only: the model was trained with --no-cls-dist (mean pooling)")
parser.add_argument("--batch-size", type=int, default=12, help="Evaluation batch size (no effect on results)")
parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
parser.add_argument("--duration-sec", type=float, default=10.0, help="Audio clip duration in seconds (must match training)")
parser.add_argument("--target-frames", type=int, default=1000, help="Spectrogram frames (must match training)")
parser.add_argument("--norm-mean", type=float, default=-4.6476, help="Normalization mean (must match training; CMKD scripts default -4.6476)")
parser.add_argument("--norm-std", type=float, default=4.5699, help="Normalization std (must match training; CMKD scripts default 4.5699)")
parser.add_argument("--output", type=str, default=None, help="Write results as JSON here (default: <checkpoint dir>/eval_<split>_<checkpoint name>.json)")
parser.add_argument("--mock", action="store_true", help="Synthetic data and a random model (smoke test); needs --model cnn|ast")
args = parser.parse_args()

if args.checkpoint is None and not args.mock:
    parser.error("--checkpoint is required unless --mock is set")
if args.mock and args.checkpoint is None and args.model == "auto":
    parser.error("--mock without --checkpoint needs --model cnn or --model ast")

NUM_CLASSES = 200

# head.weight has shape (num_classes, feature_dim): feature_dim identifies the backbone size.
AST_ARCH_BY_DIM = {
    192: {"tok_dim": 192, "num_head": 3, "num_layer": 12, "name": "DeiT ViT-Tiny"},
    384: {"tok_dim": 384, "num_head": 6, "num_layer": 12, "name": "DeiT ViT-Small"},
    768: {"tok_dim": 768, "num_head": 12, "num_layer": 12, "name": "DeiT ViT-Base"},
}
CNN_ARCH_BY_DIM = {cfg["feature_dim"]: arch for arch, cfg in CNN_ARCH_CONFIGS.items()}


def load_state_dict(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    meta = {k: checkpoint.get(k) for k in ("epoch", "best_map")} if isinstance(checkpoint, dict) else {}
    return state_dict, meta


def build_model(state_dict):
    family = args.model
    if family == "auto":
        if any(k.startswith("encoder.") for k in state_dict):
            family = "ast"
        elif any(k.startswith("features.") for k in state_dict):
            family = "cnn"
        else:
            raise ValueError("Could not auto-detect model family from checkpoint keys; pass --model cnn|ast")

    feature_dim = state_dict["head.weight"].shape[1] if state_dict is not None else None

    if family == "cnn":
        arch = CNN_ARCH_BY_DIM[feature_dim] if feature_dim is not None else "b0"
        model = CNNClassifier(arch=arch, num_classes=NUM_CLASSES, pretrained=False, dropout=0.0)
        desc = f"CNN {CNN_ARCH_CONFIGS[arch]['name']}"
    else:
        cfg = AST_ARCH_BY_DIM[feature_dim] if feature_dim is not None else AST_ARCH_BY_DIM[768]
        model = Classifer(
            tok_dim=cfg["tok_dim"],
            num_classes=NUM_CLASSES,
            c_in=1,
            overlap=6,
            patch_size=16,
            size=(128, args.target_frames),
            num_head=cfg["num_head"],
            num_layer=cfg["num_layer"],
            pretrained_dino=False,  # weights come from the checkpoint
            use_cls_dist=not args.no_cls_dist,
        )
        desc = f"AST {cfg['name']}"

    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    return model, desc


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint is not None:
        if not os.path.isfile(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        state_dict, meta = load_state_dict(args.checkpoint)
    else:
        state_dict, meta = None, {}

    model, desc = build_model(state_dict)
    model = model.to(device).eval()

    dataset = FSD50KDataset(
        root_dir=args.data_path,
        split=args.split,
        duration_sec=args.duration_sec,
        target_frames=args.target_frames,
        mock=args.mock,
        num_classes=NUM_CLASSES,
        normalize=True,
        norm_mean=args.norm_mean,
        norm_std=args.norm_std,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Evaluating {desc} from {args.checkpoint or '<random init, mock>'}")
    if meta.get("epoch") is not None:
        best_val = meta.get("best_map")
        best_desc = f", best val mAP during training {best_val:.4f}" if best_val is not None else ""
        print(f"  • Checkpoint epoch: {meta['epoch']}{best_desc}")
    print(f"  • Split: {args.split} ({len(dataset)} clips), device: {device}")

    criterion = nn.BCEWithLogitsLoss()
    metrics = MultiLabelClassificationMetrics(num_classes=NUM_CLASSES)
    running_loss = 0.0

    with torch.no_grad():
        for spectrograms, targets in tqdm(loader, desc=f"Eval ({args.split})"):
            spectrograms = spectrograms.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(spectrograms)
            logits = logits.float()
            running_loss += criterion(logits, targets.float()).item()
            metrics.update(logits, targets)

    results = {k: float(v) for k, v in metrics.compute().items()}
    results["bce"] = running_loss / max(1, len(loader))

    print("=== Results ===")
    for key in ("mAP", "mAUC", "micro_f1", "macro_f1", "top1_hit", "top5_hit", "bce"):
        if key in results:
            print(f"  {key:>9}: {results[key]:.4f}")

    output_path = args.output
    if output_path is None and args.checkpoint is not None:
        ckpt = Path(args.checkpoint)
        output_path = str(ckpt.parent / f"eval_{args.split}_{ckpt.stem}.json")
    if output_path is not None:
        payload = {
            "checkpoint": args.checkpoint,
            "model": desc,
            "split": args.split,
            "num_clips": len(dataset),
            "checkpoint_epoch": meta.get("epoch"),
            "results": results,
        }
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved results to {output_path}")


if __name__ == "__main__":
    evaluate()
