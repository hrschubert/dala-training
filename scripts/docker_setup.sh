#!/usr/bin/env bash
# =============================================================================
# Dala / lczero-training — Docker (Ubuntu + NVIDIA) Setup Script
#
# Usage:
#   ./docker_setup.sh <input.pgn> <config.textproto>
#
# Assumptions:
#   - Docker container is Ubuntu-based with NVIDIA drivers already installed
#   - Script is run as a regular user (not root) with sudo access
#   - The PGN file may be plain .pgn or compressed .pgn.zst
#
# What this script does:
#   1. Installs system packages (build tools, zstd, cmake, git, etc.)
#   2. Installs Miniconda with Python 3.13 in ~/miniconda3
#   3. Clones the dala-training repository to ~/dala-training
#   4. Builds the C++ dataloader extension
#   5. Compiles protobuf definitions
#   6. Installs the Python package in editable mode
#   7. Builds trainingdata-tool (PGN → V4 converter)
#   8. Converts the PGN to V6 training data under ~/training_data/v6
#   9. Patches the textproto config to point to the data/checkpoint/export dirs
#  10. Runs lc0-init to create the initial checkpoint
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Parse arguments
# ---------------------------------------------------------------------------
if [ $# -lt 2 ]; then
    echo "Usage: $0 <input.pgn[.zst]> <config.textproto>"
    echo ""
    echo "Arguments:"
    echo "  input.pgn[.zst]    Path to the PGN file (plain or zstd-compressed)"
    echo "  config.textproto   Path to the training configuration file"
    exit 1
fi

INPUT_PGN="$(realpath "$1")"
INPUT_CONFIG="$(realpath "$2")"

if [ ! -f "$INPUT_PGN" ]; then
    echo "Error: PGN file not found: $INPUT_PGN"; exit 1
fi
if [ ! -f "$INPUT_CONFIG" ]; then
    echo "Error: Config file not found: $INPUT_CONFIG"; exit 1
fi

# Derive a run name from the config filename (e.g. dala_1100_e.textproto → dala_1100_e)
RUN_NAME="$(basename "$INPUT_CONFIG" .textproto)"

HOME_DIR="$HOME"
REPO_DIR="${HOME_DIR}/dala-training"
CONDA_DIR="${HOME_DIR}/miniconda3"
CONDA_ENV="lczero"
TRAINING_DATA_DIR="${HOME_DIR}/training_data"
V4_DIR="${TRAINING_DATA_DIR}/v4"
V6_DIR="${TRAINING_DATA_DIR}/v6"
CHECKPOINT_DIR="${TRAINING_DATA_DIR}/${RUN_NAME}_checkpoint"
EXPORT_DIR="${TRAINING_DATA_DIR}/${RUN_NAME}_export"
TENSORBOARD_DIR="${TRAINING_DATA_DIR}/${RUN_NAME}_tensorboard"
TRAININGDATA_TOOL_DIR="${HOME_DIR}/trainingdata-tool"

mkdir -p "$TRAINING_DATA_DIR"

echo "============================================================"
echo "Dala / lczero-training Docker Setup"
echo "============================================================"
echo "  PGN input:     $INPUT_PGN"
echo "  Config:        $INPUT_CONFIG"
echo "  Run name:      $RUN_NAME"
echo "  Repository:    $REPO_DIR"
echo "  Training data: $TRAINING_DATA_DIR"
echo "============================================================"

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
echo ""
echo "=== [1/10] Installing system packages ==="
sudo apt-get update -qq
sudo apt-get install -y -qq \
    build-essential git cmake ninja-build pkg-config \
    zstd libzstd-dev zlib1g-dev \
    clang \
    libboost-dev libboost-system-dev libboost-filesystem-dev \
    wget curl ca-certificates \
    protobuf-compiler libprotobuf-dev \
    2>&1 | tail -5
echo "  Done."

# ---------------------------------------------------------------------------
# 2. Install Miniconda
# ---------------------------------------------------------------------------
echo ""
echo "=== [2/10] Installing Miniconda ==="
if [ -d "$CONDA_DIR" ]; then
    echo "  Miniconda already installed at $CONDA_DIR, skipping."
else
    wget -q "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" \
        -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
    rm /tmp/miniconda.sh
    echo "  Installed to $CONDA_DIR"
fi

# Activate conda
source "${CONDA_DIR}/etc/profile.d/conda.sh"

# Accept Conda TOS (required for non-interactive use)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# Create environment if it doesn't exist
if conda env list | grep -q "^${CONDA_ENV} "; then
    echo "  Conda env '$CONDA_ENV' already exists, activating."
else
    echo "  Creating conda env '$CONDA_ENV' with Python 3.13..."
    conda create -y -n "$CONDA_ENV" python=3.13 -q
fi
conda activate "$CONDA_ENV"
echo "  Active: $(python --version) at $(which python)"

# ---------------------------------------------------------------------------
# 3. Clone the repository
# ---------------------------------------------------------------------------
echo ""
echo "=== [3/10] Cloning dala-training repository ==="
if [ -d "$REPO_DIR" ]; then
    echo "  Repository already exists at $REPO_DIR, pulling latest."
    cd "$REPO_DIR"
    git pull --ff-only || true
else
    git clone --recurse-submodules https://github.com/hrschubert/dala-training.git "$REPO_DIR"
    cd "$REPO_DIR"
fi
# Ensure submodules are initialized (lc0)
git submodule update --init --recursive
echo "  Done."

# ---------------------------------------------------------------------------
# 4. Install Python dependencies
# ---------------------------------------------------------------------------
echo ""
echo "=== [4/10] Installing Python dependencies ==="
cd "$REPO_DIR"

# Install the package with all dependencies in editable mode
pip install -e ".[dev]" -q 2>&1 | tail -5
# Ensure grpc_tools is available for protobuf compilation
pip install grpcio-tools -q 2>&1 | tail -2

# Verify JAX can see the GPU
echo "  Checking JAX GPU access..."
python -c "
import jax
devs = jax.devices()
print(f'  JAX devices: {devs}')
if not any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devs):
    print('  WARNING: No GPU detected by JAX! Training will be slow.')
else:
    print('  GPU detected.')
"
echo "  Done."

# ---------------------------------------------------------------------------
# 5. Compile protobuf definitions
# ---------------------------------------------------------------------------
echo ""
echo "=== [5/10] Compiling protobuf definitions ==="
cd "$REPO_DIR"

python -m grpc_tools.protoc \
    --proto_path=. \
    --proto_path=libs/lc0 \
    --python_out=src/ \
    --pyi_out=src/ \
    proto/*.proto

# Also compile lc0 protos (net, hlo, onnx) needed by model_config and converter
python -m grpc_tools.protoc \
    --proto_path=libs/lc0 \
    --python_out=src/ \
    --pyi_out=src/ \
    libs/lc0/proto/net.proto \
    libs/lc0/proto/hlo.proto \
    libs/lc0/proto/onnx.proto

echo "  Done."

# ---------------------------------------------------------------------------
# 6. Build C++ dataloader extension
# ---------------------------------------------------------------------------
echo ""
echo "=== [6/10] Building C++ dataloader extension ==="
cd "$REPO_DIR"

# Create native.ini pointing to the conda python
cat > native_docker.ini << EOF
[binaries]
python = '$(which python)'
EOF

if [ -d "build/release" ]; then
    echo "  Build directory exists, reconfiguring..."
    CXX=clang++ CC=clang meson setup build/release \
        --buildtype=release --native-file=native_docker.ini --reconfigure \
        2>&1 | tail -3
else
    CXX=clang++ CC=clang meson setup build/release \
        --buildtype=release --native-file=native_docker.ini \
        2>&1 | tail -3
fi

meson compile -C build/release/ 2>&1 | tail -5

# Create symlink so Python can find the .so
SO_FILE=$(ls build/release/_lczero_training.cpython-*-x86_64-linux-gnu.so 2>/dev/null | head -1)
if [ -z "$SO_FILE" ]; then
    echo "  ERROR: C++ extension .so not found after build!"
    exit 1
fi
ln -sfT "../../${SO_FILE}" src/lczero_training/_lczero_training.so
echo "  Linked: src/lczero_training/_lczero_training.so -> $SO_FILE"
echo "  Done."

# ---------------------------------------------------------------------------
# 7. Build trainingdata-tool (PGN → V4)
# ---------------------------------------------------------------------------
echo ""
echo "=== [7/10] Building trainingdata-tool ==="
if [ -x "${TRAININGDATA_TOOL_DIR}/trainingdata-tool" ]; then
    echo "  trainingdata-tool already built, skipping."
else
    if [ ! -d "$TRAININGDATA_TOOL_DIR" ]; then
        git clone --recurse-submodules https://github.com/DanielUranga/trainingdata-tool.git "$TRAININGDATA_TOOL_DIR"
		git checkout train_on_pgn
    fi
    cd "$TRAININGDATA_TOOL_DIR"
    git submodule update --init --recursive

    # CMakeLists.txt is missing a project() declaration and has a UTF-8 BOM.
    # Strip BOM and inject project() if absent.
    sed -i '1s/^\xEF\xBB\xBF//' CMakeLists.txt
    if ! grep -q 'project(' CMakeLists.txt; then
        sed -i 's/cmake_minimum_required\s*([^)]*)/&\nproject(trainingdata-tool LANGUAGES CXX C)/' CMakeLists.txt
        echo "  Patched CMakeLists.txt: stripped BOM + added project() declaration"
    fi

    # Clean any previous failed build
    rm -rf build
    mkdir -p build && cd build
    # GCC 13+ needs explicit cstdint/cstring includes
    cmake .. -DCMAKE_CXX_FLAGS='-include cstdint -include cstring'
    make -j"$(nproc)"
    # Copy binary to parent dir for easy access
    cp src/trainingdata-tool "${TRAININGDATA_TOOL_DIR}/trainingdata-tool" 2>/dev/null || \
    cp trainingdata-tool "${TRAININGDATA_TOOL_DIR}/trainingdata-tool" 2>/dev/null || true
    if [ ! -x "${TRAININGDATA_TOOL_DIR}/trainingdata-tool" ]; then
        # Binary might be directly in build/
        find . -name trainingdata-tool -type f -executable -exec cp {} "${TRAININGDATA_TOOL_DIR}/trainingdata-tool" \;
    fi
    echo "  Built at ${TRAININGDATA_TOOL_DIR}/trainingdata-tool"
fi
echo "  Done."

# ---------------------------------------------------------------------------
# 8. Convert PGN → V4 → V6 training data
# ---------------------------------------------------------------------------
echo ""
echo "=== [8/10] Converting PGN to V6 training data ==="
mkdir -p "$V4_DIR" "$V6_DIR"
cd "$REPO_DIR"

if [ "$(find "$V6_DIR" -name '*.gz' 2>/dev/null | wc -l)" -gt 0 ]; then
    V6_COUNT=$(find "$V6_DIR" -name '*.gz' | wc -l)
    echo "  V6 directory already contains $V6_COUNT files, skipping conversion."
    echo "  (Delete $V6_DIR to reconvert.)"
else
    # Step 8a: Decompress if .zst
    if [[ "$INPUT_PGN" == *.zst ]]; then
        echo "  Decompressing $(basename "$INPUT_PGN")..."
        PGN_PLAIN="${TRAINING_DATA_DIR}/input.pgn"
        zstd -d "$INPUT_PGN" -o "$PGN_PLAIN" --no-progress -f
        echo "  Decompressed to $(du -h "$PGN_PLAIN" | cut -f1)"
    else
        PGN_PLAIN="$INPUT_PGN"
    fi

    # Step 8b: PGN → V4 (parallel)
    echo "  Converting PGN to V4 training data (parallel)..."
    export TRAININGDATA_TOOL="${TRAININGDATA_TOOL_DIR}/trainingdata-tool"
    if [ ! -x "$TRAININGDATA_TOOL" ]; then
        echo "  ERROR: trainingdata-tool not found at $TRAININGDATA_TOOL"
        exit 1
    fi
    bash scripts/parallel_pgn_convert.sh "$PGN_PLAIN" "$V4_DIR"
    N_V4=$(find "$V4_DIR" -name '*.gz' | wc -l)
    echo "  Generated $N_V4 V4 chunk files"

    # Step 8c: V4 → V6 (parallel, defaults to all CPU cores)
    echo "  Converting V4 to V6 training data (parallel)..."
    python scripts/v4_to_v6.py "$V4_DIR" "$V6_DIR"
    N_V6=$(find "$V6_DIR" -name '*.gz' | wc -l)
    echo "  Generated $N_V6 V6 chunk files"
fi
echo "  Done."

# ---------------------------------------------------------------------------
# 9. Patch textproto config with correct paths
# ---------------------------------------------------------------------------
echo ""
echo "=== [9/10] Preparing configuration ==="
cd "$REPO_DIR"
mkdir -p configs
CONFIG_DEST="configs/${RUN_NAME}.textproto"
cp "$INPUT_CONFIG" "$CONFIG_DEST"

# Count V6 chunk files for chunk_pool_size
N_V6=$(find "$V6_DIR" -name '*.gz' | wc -l)

# Patch directory paths in the config to match this environment
python3 << PYEOF
import re, sys

config_path = "${CONFIG_DEST}"
with open(config_path, "r") as f:
    text = f.read()

# Replace data directory
text = re.sub(
    r'(directory:\s*")[^"]*(")',
    r'\g<1>${V6_DIR}\2',
    text,
)

# Replace chunk_pool_size to match actual file count
text = re.sub(
    r'(chunk_pool_size:\s*)\d+',
    r'\g<1>${N_V6}',
    text,
)

# Replace checkpoint path
text = re.sub(
    r'(path:\s*")[^"]*checkpoint[^"]*(")',
    r'\g<1>${CHECKPOINT_DIR}\2',
    text,
)

# Replace tensorboard path
text = re.sub(
    r'(tensorboard_path:\s*")[^"]*(")',
    r'\g<1>${TENSORBOARD_DIR}\2',
    text,
)

# Replace export path (keep the filename pattern)
text = re.sub(
    r'(destination_filename:\s*")[^"]*(/[^/"]+")' ,
    r'\g<1>${EXPORT_DIR}\2',
    text,
)

with open(config_path, "w") as f:
    f.write(text)

print(f"  Config written to {config_path}")
print(f"    data dir:       ${V6_DIR}")
print(f"    chunk_pool:     ${N_V6}")
print(f"    checkpoint:     ${CHECKPOINT_DIR}")
print(f"    tensorboard:    ${TENSORBOARD_DIR}")
print(f"    export:         ${EXPORT_DIR}")
PYEOF
echo "  Done."

# ---------------------------------------------------------------------------
# 10. Initialize training checkpoint
# ---------------------------------------------------------------------------
echo ""
echo "=== [10/10] Initializing training (lc0-init) ==="
cd "$REPO_DIR"

mkdir -p "$CHECKPOINT_DIR" "$EXPORT_DIR" "$TENSORBOARD_DIR"

if [ -d "${CHECKPOINT_DIR}/0" ]; then
    echo "  Checkpoint already exists at $CHECKPOINT_DIR, skipping init."
    echo "  (Delete $CHECKPOINT_DIR to reinitialize.)"
else
    echo "  Running: lc0-init --config $CONFIG_DEST"
    echo "  (working dir: $(pwd))"
    lc0-init --config "$CONFIG_DEST" 2>&1
    if [ -d "${CHECKPOINT_DIR}/0" ]; then
        echo "  Checkpoint initialized successfully."
    else
        echo "  ERROR: lc0-init completed but no checkpoint was created at ${CHECKPOINT_DIR}/0"
        echo "  Check the config file paths:"
        grep -E 'path:|directory:' "$CONFIG_DEST"
        exit 1
    fi
fi
echo "  Done."

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  SETUP COMPLETE"
echo "============================================================"
echo ""
echo "  Repository:     $REPO_DIR"
echo "  Conda env:      $CONDA_ENV (Python $(python --version 2>&1 | cut -d' ' -f2))"
echo "  Training data:  $V6_DIR ($N_V6 V6 chunk files)"
echo "  Checkpoint:     $CHECKPOINT_DIR"
echo "  Config:         $REPO_DIR/$CONFIG_DEST"
echo ""
echo "  To start training:"
echo "    source ${CONDA_DIR}/etc/profile.d/conda.sh"
echo "    conda activate $CONDA_ENV"
echo "    cd $REPO_DIR"
echo "    nohup lc0-train --config $CONFIG_DEST > ~/train.log 2>&1 &"
echo ""
echo "  To monitor:"
echo "    tail -f ~/train.log"
echo "    tensorboard --logdir $TENSORBOARD_DIR --bind_all"
echo ""
echo "  To evaluate:"
echo "    lc0-eval --config $CONFIG_DEST --num-samples 100"
echo ""
echo "============================================================"
