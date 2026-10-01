#!/usr/bin/env bash
# Quick checks (~5 min, no dataset or teacher needed). Run before the long jobs.
#   bash scripts/aws/quick_check.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SMOKE_DIR="${SMOKE_DIR:-/tmp/cmkd_smoke}"
rm -rf "$SMOKE_DIR"
print_settings "CMKD quick checks"

echo "-- [1/5] environment"
python -c "import torch, torchaudio, torchvision; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"

echo "-- [2/5] KD loss sanity checks"
python losses/distillation.py

echo "-- [3/5] CNN teacher smoke test (synthetic data, 2 epochs)"
python scripts/train_cnn_classifier.py --mock --num-epoch 2 --num-workers 0 \
    --checkpoint-dir "$SMOKE_DIR/cnn/"

echo "-- [4/5] KD smoke test (synthetic data, random teacher, 2 epochs)"
python scripts/train_kd.py --mock --num-epoch 2 --num-workers 0 --grad-accum-steps 2 \
    --checkpoint-dir "$SMOKE_DIR/kd/"

echo "-- [5/5] evaluation script on both smoke checkpoints"
python scripts/evaluate.py --mock --checkpoint "$SMOKE_DIR/cnn/best_cnn.pth" --num-workers 0
python scripts/evaluate.py --mock --checkpoint "$SMOKE_DIR/kd/best_kd_ast.pth" --num-workers 0

for f in "$SMOKE_DIR/cnn/metrics.csv" "$SMOKE_DIR/kd/metrics.csv"; do
    [[ -f "$f" ]] || { echo "ERROR: expected $f to exist" >&2; exit 1; }
done

if [[ -d "$DATA_PATH" ]]; then
    echo "-- dataset found at $DATA_PATH"
else
    echo "-- WARNING: DATA_PATH '$DATA_PATH' not found yet (needed for run_cnn.sh / run_kd.sh)"
fi

echo "ALL QUICK CHECKS PASSED"
