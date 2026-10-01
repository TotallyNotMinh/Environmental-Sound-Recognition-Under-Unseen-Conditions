# Shared settings for the AWS launch scripts. Sourced, not executed.
# Override any value by exporting it before running a script, e.g.
#   DATA_PATH=/mnt/fsd50k NUM_WORKERS=15 bash scripts/aws/run_kd.sh

set -euo pipefail

# Run from the repo root regardless of where the script was called from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DATA_PATH="${DATA_PATH:-/data/fsd50k}"      # folder that contains FSD50K.ground_truth/
RUNS_DIR="${RUNS_DIR:-/data/runs}"          # checkpoints, logs, metrics.csv, eval results

# GPUs: default = all visible GPUs.
if [[ -z "${NGPU:-}" ]]; then
    NGPU="$( (nvidia-smi -L 2>/dev/null || true) | wc -l | tr -d ' ')"
    [[ "$NGPU" -ge 1 ]] || NGPU=1
fi

# DataLoader workers PER GPU process: (vCPUs - 1) split across GPU processes.
if [[ -z "${NUM_WORKERS:-}" ]]; then
    NUM_WORKERS=$(( ( $(nproc) - 1 ) / NGPU ))
    [[ "$NUM_WORKERS" -ge 1 ]] || NUM_WORKERS=1
fi

# Unbuffered Python so `tee` writes progress to the log immediately.
export PYTHONUNBUFFERED=1

# 1 GPU -> plain python; several GPUs -> torchrun (the training scripts detect it via RANK/WORLD_SIZE).
if [[ "$NGPU" -gt 1 ]]; then
    LAUNCH=(torchrun --standalone --nproc_per_node="$NGPU")
else
    LAUNCH=(python)
fi

check_data_path() {
    if [[ ! -d "$DATA_PATH" ]]; then
        echo "ERROR: DATA_PATH '$DATA_PATH' does not exist. Set DATA_PATH to the folder containing FSD50K.ground_truth/." >&2
        exit 1
    fi
}

print_settings() {
    echo "== $1 =="
    echo "  repo:        $REPO_ROOT ($(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?') @ $(git rev-parse --short HEAD 2>/dev/null || echo '?'))"
    echo "  data:        $DATA_PATH"
    echo "  runs:        $RUNS_DIR"
    echo "  GPUs:        $NGPU   workers/GPU: $NUM_WORKERS   launcher: ${LAUNCH[*]}"
}
