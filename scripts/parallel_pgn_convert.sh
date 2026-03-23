#!/usr/bin/env bash
# =============================================================================
# Parallel PGN → V4 conversion using trainingdata-tool
#
# Splits a PGN file into N chunks using python-chess for correct game boundary
# detection, then runs trainingdata-tool on each chunk in parallel.
#
# Requires: python3 with chess module (pip install python-chess)
#
# Usage:
#   ./parallel_pgn_convert.sh <input.pgn> <output_v4_dir> [num_workers]
#
# Default workers = number of CPU cores (capped at 128).
# =============================================================================
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <input.pgn> <output_v4_dir> [num_workers]"
    exit 1
fi

INPUT_PGN="$(realpath "$1")"
OUTPUT_DIR="$(realpath -m "$2")"
_NPROC=$(nproc)
_MAX_WORKERS=$(( _NPROC < 128 ? _NPROC : 128 ))
NUM_WORKERS="${3:-$_MAX_WORKERS}"
TOOL="${TRAININGDATA_TOOL:-$(dirname "$0")/../../trainingdata-tool/trainingdata-tool}"

if [ ! -f "$INPUT_PGN" ]; then
    echo "Error: PGN file not found: $INPUT_PGN"; exit 1
fi
if [ ! -x "$TOOL" ]; then
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
# Step 1: Split PGN into chunks using python-chess for correct parsing
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 1: Splitting PGN into chunks using python-chess ==="

export INPUT_PGN FILE_SIZE NUM_WORKERS TMPDIR

python3 << 'PYEOF'
import os, sys

try:
    import chess.pgn
except ImportError:
    print("ERROR: python-chess not installed. Run: pip install python-chess", file=sys.stderr)
    sys.exit(1)

pgn_path = os.environ["INPUT_PGN"]
file_size = int(os.environ["FILE_SIZE"])
num_workers = int(os.environ["NUM_WORKERS"])
tmpdir = os.environ["TMPDIR"]

chunk_size = file_size // num_workers

def find_game_boundary_with_chess(pgn_path, approx_offset):
    """Use python-chess to find a verified game boundary near approx_offset.

    Seeks to approx_offset, scans forward for a line starting with '[Event ',
    then uses chess.pgn.read_game() to verify it's a real game start.
    Returns the byte offset of the verified game start, or None.
    """
    with open(pgn_path, "r", encoding="utf-8", errors="replace") as f:
        # Seek to approximate position
        f.seek(approx_offset)

        # Read and discard the rest of the current line (we may be mid-line)
        f.readline()

        # Now scan forward for the pattern: blank line(s) followed by [Event
        # We look for a line that starts with [Event after whitespace-only lines
        max_scan = 10 * 1024 * 1024  # Scan up to 10MB forward
        scanned = 0
        prev_was_blank = False

        while scanned < max_scan:
            pos_before = f.tell()
            line = f.readline()
            if not line:
                return None  # EOF
            scanned += len(line.encode("utf-8", errors="replace"))

            stripped = line.strip()
            if stripped == "":
                prev_was_blank = True
                continue

            if prev_was_blank and stripped.startswith("[Event "):
                # Found a candidate. Verify with python-chess by trying to
                # read a game from this position.
                candidate_offset = pos_before
                f.seek(candidate_offset)
                try:
                    game = chess.pgn.read_game(f)
                    if game is not None:
                        # Successfully parsed a game — this is a real boundary
                        return candidate_offset
                except Exception:
                    pass
                # If parsing failed, continue scanning from after this line
                f.seek(pos_before + len(line.encode("utf-8", errors="replace")))

            prev_was_blank = False

        return None


# Find split points
split_offsets = [0]

for i in range(1, num_workers):
    target = chunk_size * i
    boundary = find_game_boundary_with_chess(pgn_path, target)
    if boundary is not None and boundary > split_offsets[-1]:
        split_offsets.append(boundary)
        print(f"  Split point {i}/{num_workers-1}: offset {boundary:,} "
              f"({boundary/file_size*100:.1f}%)")
    else:
        print(f"  Split point {i}/{num_workers-1}: not found, merging with previous chunk")

split_offsets.append(file_size)

num_chunks = len(split_offsets) - 1
print(f"  Split into {num_chunks} chunks")

with open(f"{tmpdir}/splits.txt", "w") as out:
    for i in range(num_chunks):
        start = split_offsets[i]
        end = split_offsets[i + 1]
        size = end - start
        print(f"    Chunk {i}: offset {start:,}, size {size/1e9:.2f} GB")
        out.write(f"{start} {end}\n")

# Write chunk files in parallel using multiprocessing
import time
from multiprocessing import Pool, Value, Lock
import multiprocessing

print(f"\n  Writing chunk files with python-chess in parallel ({num_workers} workers)...")

def write_chunk(args):
    """Write a single chunk PGN file using python-chess. Runs in a worker process."""
    chunk_idx, start, end, pgn_path, tmpdir = args
    import chess.pgn, time, os
    chunk_path = f"{tmpdir}/chunk_{chunk_idx}.pgn"
    games_written = 0
    chunk_start = time.time()

    with open(pgn_path, "r", encoding="utf-8", errors="replace") as f:
        f.seek(start)
        with open(chunk_path, "w", encoding="utf-8") as out:
            while f.tell() < end:
                try:
                    game = chess.pgn.read_game(f)
                except Exception:
                    continue
                if game is None:
                    break
                print(game, file=out, end="\n\n")
                games_written += 1

    chunk_mb = os.path.getsize(chunk_path) / 1e6
    elapsed = time.time() - chunk_start
    return chunk_idx, games_written, chunk_mb, elapsed

# Build task list
tasks = []
for i in range(num_chunks):
    tasks.append((i, split_offsets[i], split_offsets[i + 1], pgn_path, tmpdir))

write_start = time.time()
total_games = 0
chunks_done = 0

# Use min(num_workers, num_chunks) processes — no point spawning more than chunks
pool_size = min(num_workers, num_chunks)
with Pool(pool_size) as pool:
    for chunk_idx, games_written, chunk_mb, elapsed in pool.imap_unordered(write_chunk, tasks):
        total_games += games_written
        chunks_done += 1
        print(f"    Chunk {chunk_idx+1}/{num_chunks}: {games_written:,} games, "
              f"{chunk_mb:.1f} MB, {elapsed:.1f}s  "
              f"[{chunks_done}/{num_chunks} chunks done]")

overall_elapsed = time.time() - write_start
rate = total_games / overall_elapsed if overall_elapsed > 0 else 0
print(f"  Total: {total_games:,} games in {overall_elapsed:.0f}s "
      f"({rate:.0f} games/s)")

# Rewrite splits.txt to signal that chunks are pre-extracted (no dd needed)
with open(f"{tmpdir}/splits.txt", "w") as out:
    for i in range(num_chunks):
        out.write(f"{i}\n")

print("  All chunks written successfully.")
PYEOF

NUM_CHUNKS=$(wc -l < "$TMPDIR/splits.txt")
echo "  Created $NUM_CHUNKS chunks"

# ---------------------------------------------------------------------------
# Step 2: Run trainingdata-tool on each chunk in parallel
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 2: Converting chunks in parallel ==="

convert_chunk() {
    local chunk_idx="$1"
    local chunk_pgn="${TMPDIR}/chunk_${chunk_idx}.pgn"
    local out_dir="${OUTPUT_DIR}/chunk_${chunk_idx}"

    if [ ! -f "$chunk_pgn" ]; then
        echo "  [Chunk $chunk_idx/$NUM_CHUNKS] WARNING: chunk file missing, skipping"
        return
    fi

    local chunk_size_h
    chunk_size_h=$(numfmt --to=iec "$(stat -c%s "$chunk_pgn")" 2>/dev/null || echo "?")

    mkdir -p "$out_dir"
    echo "  [Chunk $chunk_idx/$NUM_CHUNKS] Converting ${chunk_size_h}..."

    cd "$out_dir"
    "$TOOL" "$chunk_pgn" 2>&1 | while IFS= read -r line; do
        echo "  [Chunk $chunk_idx/$NUM_CHUNKS] $line"
    done

    # Clean up chunk PGN to save disk space
    rm -f "$chunk_pgn"

    local n_files
    n_files=$(find "$out_dir" -name '*.gz' | wc -l)
    echo "  [Chunk $chunk_idx/$NUM_CHUNKS] DONE — $n_files V4 files"
}

export -f convert_chunk
export INPUT_PGN TOOL TMPDIR OUTPUT_DIR NUM_CHUNKS

START_TIME=$(date +%s)

chunk_idx=0
pids=()
while IFS= read -r idx; do
    convert_chunk "$idx" &
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
