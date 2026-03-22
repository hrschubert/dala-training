#!/usr/bin/env bash
# =============================================================================
# Start lczero training after docker_setup.sh has completed.
#
# Usage:
#   ./docker_train.sh <config.textproto>
#
# Optional environment variables:
#   EXPORT_EVERY      - export network every N steps (default: 0 = only at end)
#   BACKGROUND        - set to 1 to run in background with nohup (default: 0)
# =============================================================================

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <config.textproto>"
    echo ""
    echo "  config.textproto   The run name or path to the textproto config."
    echo "                     If just a name (e.g. dala_1600_a), looks in"
    echo "                     ~/dala-training/configs/<name>.textproto"
    echo ""
    echo "Optional env vars:"
    echo "  EXPORT_EVERY=0         Export network every N steps (0=end only)"
    echo "  BACKGROUND=1           Run in background with nohup"
    exit 1
fi

# ---------------------------------------------------------------------------
# Resolve config path
# ---------------------------------------------------------------------------
INPUT="$1"
HOME_DIR="$HOME"
REPO_DIR="${HOME_DIR}/dala-training"
CONDA_DIR="${HOME_DIR}/miniconda3"
CONDA_ENV="lczero"

EXPORT_EVERY="${EXPORT_EVERY:-0}"
BACKGROUND="${BACKGROUND:-0}"

# If input is just a name (no slashes, no .textproto), resolve it
if [[ "$INPUT" != */* ]] && [[ "$INPUT" != *.textproto ]]; then
    CONFIG_PATH="${REPO_DIR}/configs/${INPUT}.textproto"
elif [[ "$INPUT" == *.textproto ]] && [[ "$INPUT" != */* ]]; then
    CONFIG_PATH="${REPO_DIR}/configs/${INPUT}"
else
    CONFIG_PATH="$(realpath "$INPUT")"
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config not found: $CONFIG_PATH"
    exit 1
fi

RUN_NAME="$(basename "$CONFIG_PATH" .textproto)"
LOG_FILE="${HOME_DIR}/train_${RUN_NAME}.log"

echo "============================================================"
echo "  Dala Training — ${RUN_NAME}"
echo "============================================================"
echo "  Config:           $CONFIG_PATH"
echo "  Export every:      $EXPORT_EVERY steps (0=end only)"
echo "  Log file:         $LOG_FILE"
echo "============================================================"

# ---------------------------------------------------------------------------
# Activate conda
# ---------------------------------------------------------------------------
source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$REPO_DIR"

echo ""
echo "  Python: $(which python) ($(python --version 2>&1))"
echo "  JAX devices: $(python -c 'import jax; print(jax.devices())' 2>/dev/null)"
echo ""

# ---------------------------------------------------------------------------
# Build train command
# ---------------------------------------------------------------------------
TRAIN_CMD="python src/lczero_training/commands/train.py \
    --config $CONFIG_PATH"

if [ "$EXPORT_EVERY" -gt 0 ]; then
    TRAIN_CMD="$TRAIN_CMD --export-every $EXPORT_EVERY"
fi

# ---------------------------------------------------------------------------
# Start training
# ---------------------------------------------------------------------------
if [ "$BACKGROUND" = "1" ]; then
    echo "Starting training in background..."
    echo "  Log: tail -f $LOG_FILE"
    echo ""
    nohup bash -c "$TRAIN_CMD" > "$LOG_FILE" 2>&1 &
    TRAIN_PID=$!
    echo "  PID: $TRAIN_PID"
    echo "  To stop: kill $TRAIN_PID"
    echo ""
    # Wait a few seconds and check it didn't crash immediately
    sleep 5
    if kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo "  Training is running. Last log lines:"
        tail -5 "$LOG_FILE" 2>/dev/null || true
    else
        echo "  ERROR: Training exited immediately. Last log lines:"
        tail -20 "$LOG_FILE" 2>/dev/null || true
        exit 1
    fi
else
    echo "Starting training (foreground)..."
    echo "Press Ctrl+C to stop (checkpoint will be saved on next step boundary)."
    echo ""
    bash -c "$TRAIN_CMD" 2>&1 | tee "$LOG_FILE"
fi
