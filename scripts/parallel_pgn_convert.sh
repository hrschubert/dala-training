#!/usr/bin/env bash
# =============================================================================
# Parallel PGN → V4 conversion using trainingdata-tool
#
# Splits a PGN file into N chunks at game boundaries, then runs
# trainingdata-tool on each chunk in parallel.
#
# Usage:
#   ./parallel_pgn_convert.sh <input.pgn> <output_v4_dir> [num_workers]
#
# Default workers = number of CPU cores.
# =============================================================================
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <input.pgn> <output_v4_dir> [num_workers]"
    exit 1
fi

INPUT_PGN="$(realpath "$1")"
OUTPUT_DIR="$(realpath -m "$2")"
# Cap at 32 workers — more causes disk/memory thrashing
_NPROC=$(nproc)
_MAX_WORKERS=$(( _NPROC < 32 ? _NPROC : 32 ))
NUM_WORKERS="${3:-$_MAX_WORKERS}"
TOOL="${TRAININGDATA_TOOL:-$(dirname "$0")/../../trainingdata-tool/trainingdata-tool}"

if [ ! -f "$INPUT_PGN" ]; then
    echo "Error: PGN file not found: $INPUT_PGN"; exit 1
fi
if [ ! -x "$TOOL" ]; then
    # Try common locations
    for candidate in \
        "${HOME}/trainingdata-tool/trainingdata-tool" \
        "$(dirname "$0")/../../trainingdata-tool/trainingdata-tool"; do
        if [ -x "$candidate" ]; then
            TOOL="$candidate"
            break
        fi
    done
    if [ ! -x "$TOOL" ]; then
        echo "Error: trainingdata-tool not found. Set TRAININGDATA_TOOL env var."
        exit 1
    fi
fi

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT
mkdir -p "$OUTPUT_DIR"

FILE_SIZE=$(stat -c%s "$INPUT_PGN" 2>/dev/null || stat -f%z "$INPUT_PGN")
echo "============================================================"
echo "Parallel PGN → V4 Conversion"
echo "============================================================"
echo "  Input:   $INPUT_PGN ($(numfmt --to=iec "$FILE_SIZE" 2>/dev/null || echo "${FILE_SIZE} bytes"))"
echo "  Output:  $OUTPUT_DIR"
echo "  Workers: $NUM_WORKERS"
echo "  Tool:    $TOOL"
echo "============================================================"

# ---------------------------------------------------------------------------
# Step 1: Calculate split points at game boundaries
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 1: Calculating split points ==="

# We need to split the file into NUM_WORKERS roughly equal parts,
# but only at lines starting with [Event (game boundaries).
# Strategy: calculate target byte offsets, then seek to each offset
# and scan forward to the next [Event line.

CHUNK_SIZE=$(( FILE_SIZE / NUM_WORKERS ))

# Export variables for the Python subprocess
export INPUT_PGN FILE_SIZE NUM_WORKERS TMPDIR

# Find split byte offsets (at game boundaries)
python3 << 'PYEOF'
import sys, os

pgn_path = os.environ["INPUT_PGN"]
file_size = int(os.environ["FILE_SIZE"])
num_workers = int(os.environ["NUM_WORKERS"])
tmpdir = os.environ["TMPDIR"]
chunk_size = file_size // num_workers

split_offsets = [0]

with open(pgn_path, "rb") as f:
    for i in range(1, num_workers):
        target = chunk_size * i
        # Seek to target and scan forward for a game boundary
        f.seek(target)
        # Read ahead to find next [Event line
        # Read in chunks to avoid loading too much into memory
        buf = b""
        while True:
            block = f.read(1024 * 1024)  # 1MB at a time
            if not block:
                break
            buf += block
            # Look for \n[Event  (game boundary)
            idx = buf.find(b"\n[Event ")
            if idx >= 0:
                split_offsets.append(target + idx + 1)  # +1 to skip the \n
                break
            # Keep only tail to avoid memory buildup
            if len(buf) > 2 * 1024 * 1024:
                target += len(buf) - 1024 * 1024
                buf = buf[-1024 * 1024:]

split_offsets.append(file_size)

# Write split info
with open(f"{tmpdir}/splits.txt", "w") as out:
    for i in range(len(split_offsets) - 1):
        out.write(f"{split_offsets[i]} {split_offsets[i+1]}\n")

print(f"  Split into {len(split_offsets)-1} chunks")
for i in range(len(split_offsets) - 1):
    size = split_offsets[i+1] - split_offsets[i]
    print(f"    Chunk {i}: offset {split_offsets[i]:,}, size {size/1e9:.2f} GB")
PYEOF

NUM_CHUNKS=$(wc -l < "$TMPDIR/splits.txt")
echo "  Created $NUM_CHUNKS split points"

# ---------------------------------------------------------------------------
# Step 2: Extract chunks and run trainingdata-tool in parallel
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 2: Converting chunks in parallel ==="

convert_chunk() {
    local chunk_idx="$1"
    local start_offset="$2"
    local end_offset="$3"
    local chunk_size=$((end_offset - start_offset))
    local chunk_dir="${TMPDIR}/chunk_${chunk_idx}"
    local chunk_pgn="${chunk_dir}/games.pgn"
    local out_dir="${OUTPUT_DIR}/chunk_${chunk_idx}"
    local chunk_size_h=$(numfmt --to=iec "$chunk_size" 2>/dev/null || echo "${chunk_size}")

    mkdir -p "$chunk_dir" "$out_dir"

    echo "  [Chunk $chunk_idx/$NUM_WORKERS] Extracting ${chunk_size_h}..."

    # Extract chunk using dd
    dd if="$INPUT_PGN" bs=1M iflag=skip_bytes,count_bytes \
        skip="$start_offset" count="$chunk_size" \
        of="$chunk_pgn" 2>/dev/null

    echo "  [Chunk $chunk_idx/$NUM_WORKERS] Converting games..."

    # Run trainingdata-tool, forwarding periodic progress lines
    cd "$out_dir"
    "$TOOL" "$chunk_pgn" 2>&1 | while IFS= read -r line; do
        echo "  [Chunk $chunk_idx/$NUM_WORKERS] $line"
    done

    # Clean up the extracted chunk PGN to save disk space
    rm -f "$chunk_pgn"

    local n_files
    n_files=$(find "$out_dir" -name '*.gz' | wc -l)
    echo "  [Chunk $chunk_idx/$NUM_WORKERS] DONE — $n_files V4 files"
}

export -f convert_chunk
export INPUT_PGN TOOL TMPDIR OUTPUT_DIR NUM_WORKERS

START_TIME=$(date +%s)

# Run chunks in parallel using background jobs
chunk_idx=0
pids=()
while IFS=' ' read -r start end; do
    convert_chunk "$chunk_idx" "$start" "$end" &
    pids+=($!)
    chunk_idx=$((chunk_idx + 1))
    # Limit concurrent jobs
    while [ "$(jobs -rp | wc -l)" -ge "$NUM_WORKERS" ]; do
        wait -n 2>/dev/null || true
    done
done < "$TMPDIR/splits.txt"

# Wait for all remaining jobs
echo "  All chunks dispatched, waiting for remaining jobs to finish..."
for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null || true
done
echo "  All chunks complete."

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# ---------------------------------------------------------------------------
# Step 3: Consolidate output
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 3: Consolidating output ==="

# Flatten all supervised-* dirs from chunks into a single output
# Rename files to avoid collisions
COUNTER=0
for chunk_dir in "$OUTPUT_DIR"/chunk_*/supervised-*; do
    if [ ! -d "$chunk_dir" ]; then continue; fi
    for gz_file in "$chunk_dir"/*.gz; do
        if [ ! -f "$gz_file" ]; then continue; fi
        DEST=$(printf "%s/game_%06d.gz" "$OUTPUT_DIR" "$COUNTER")
        mv "$gz_file" "$DEST"
        COUNTER=$((COUNTER + 1))
    done
done

# Clean up empty chunk directories
rm -rf "$OUTPUT_DIR"/chunk_*

TOTAL_FILES=$(find "$OUTPUT_DIR" -maxdepth 1 -name '*.gz' | wc -l)

echo ""
echo "============================================================"
echo "  CONVERSION COMPLETE"
echo "============================================================"
echo "  Output:     $OUTPUT_DIR"
echo "  V4 files:   $TOTAL_FILES"
echo "  Time:       ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "  Throughput: $(echo "scale=2; $FILE_SIZE / 1073741824 / $ELAPSED * 3600" | bc) GB/h"
echo "============================================================"
