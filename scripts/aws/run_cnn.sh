#!/usr/bin/env bash
# STEP 1: train the CNN teacher (EfficientNet-B0) WITHOUT KD, then score it on the FSD50K eval split.
# CMKD FSD50K CNN recipe: LR 5e-4, total batch 24, 50 epochs, BCE.
#   bash scripts/aws/run_cnn.sh                 # fresh run, or resume after an interruption (same command)
#   bash scripts/aws/run_cnn.sh --num-epoch 1   # extra args are passed to the training script
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RUN_DIR="$RUNS_DIR/fsd50k_cnn_b0"
mkdir -p "$RUN_DIR"
check_data_path
print_settings "STEP 1: CNN teacher -> $RUN_DIR" | tee -a "$RUN_DIR/train.log"

# --checkpoint-path points at checkpoint.pth: missing on the first run (fresh start),
# present after an interruption (resumes from the last finished epoch).
"${LAUNCH[@]}" scripts/train_cnn_classifier.py \
    --data-path "$DATA_PATH" \
    --batch-size 24 --lr 5e-4 --num-epoch 50 \
    --num-workers "$NUM_WORKERS" \
    --checkpoint-dir "$RUN_DIR/" \
    --checkpoint-path "$RUN_DIR/checkpoint.pth" \
    "$@" 2>&1 | tee -a "$RUN_DIR/train.log"

echo "== Evaluating best teacher on the FSD50K eval split ==" | tee -a "$RUN_DIR/train.log"
python scripts/evaluate.py \
    --checkpoint "$RUN_DIR/best_cnn.pth" \
    --data-path "$DATA_PATH" --split eval \
    --num-workers "$NUM_WORKERS" 2>&1 | tee -a "$RUN_DIR/train.log"

echo "STEP 1 done. Teacher for STEP 2: $RUN_DIR/best_cnn.pth"
