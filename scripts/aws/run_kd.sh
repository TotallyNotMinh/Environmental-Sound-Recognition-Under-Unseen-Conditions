#!/usr/bin/env bash
# STEP 2: distill the frozen CNN teacher into an AST-Base student, then score it on the FSD50K eval split.
# CMKD FSD50K AST recipe: LR 5e-5, total batch 12, 50 epochs, BCE + KD (lambda 0.5, tau 1.0).
#   bash scripts/aws/run_kd.sh                 # fresh run, or resume after an interruption (same command)
#   KD_ACCUM=3 bash scripts/aws/run_kd.sh      # smaller micro-batch if CUDA runs out of memory
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TEACHER="${TEACHER:-$RUNS_DIR/fsd50k_cnn_b0/best_cnn.pth}"
RUN_DIR="$RUNS_DIR/fsd50k_kd_ast_base"

# Gradient accumulation keeps the effective batch at 12 while limiting clips per GPU:
# 1 GPU -> 2 steps x 6 clips; 2+ GPUs -> 1 step (12/NGPU clips each). AST has no BatchNorm,
# so this is mathematically identical to a real batch of 12.
if [[ -z "${KD_ACCUM:-}" ]]; then
    if [[ "$NGPU" -eq 1 ]]; then KD_ACCUM=2; else KD_ACCUM=1; fi
fi

if [[ ! -f "$TEACHER" ]]; then
    echo "ERROR: teacher checkpoint '$TEACHER' not found. Run scripts/aws/run_cnn.sh first (or set TEACHER=...)." >&2
    exit 1
fi
mkdir -p "$RUN_DIR"
check_data_path
{
    print_settings "STEP 2: KD CNN -> AST-Base -> $RUN_DIR"
    echo "  teacher:     $TEACHER"
    echo "  grad accum:  $KD_ACCUM  (clips per GPU per micro-batch: $(( 12 / (NGPU * KD_ACCUM) )))"
} | tee -a "$RUN_DIR/train.log"

"${LAUNCH[@]}" scripts/train_kd.py \
    --data-path "$DATA_PATH" \
    --teacher-checkpoint "$TEACHER" \
    --batch-size 12 --grad-accum-steps "$KD_ACCUM" --lr 5e-5 --num-epoch 50 \
    --num-workers "$NUM_WORKERS" \
    --checkpoint-dir "$RUN_DIR/" \
    --checkpoint-path "$RUN_DIR/checkpoint.pth" \
    "$@" 2>&1 | tee -a "$RUN_DIR/train.log"

echo "== Evaluating best KD student on the FSD50K eval split ==" | tee -a "$RUN_DIR/train.log"
python scripts/evaluate.py \
    --checkpoint "$RUN_DIR/best_kd_ast.pth" \
    --data-path "$DATA_PATH" --split eval \
    --num-workers "$NUM_WORKERS" 2>&1 | tee -a "$RUN_DIR/train.log"

echo "STEP 2 done. Paper target (CMKD, EfficientNet-B0 -> AST-Base, FSD50K eval): ~0.617 mAP"
