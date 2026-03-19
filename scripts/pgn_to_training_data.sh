#!/bin/bash
# Convert .pgn.zst files to V6 training data for lczero-training.
#
# Usage:
#   ./pgn_to_training_data.sh input.pgn.zst output_dir/
#   ./pgn_to_training_data.sh --lichess input.pgn.zst output_dir/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TOOL="${TRAININGDATA_TOOL:-../../trainingdata-tool/trainingdata-tool}"
V4_TO_V6="${SCRIPT_DIR}/v4_to_v6.py"

LICHESS_FLAG=""
if [ "${1:-}" = "--lichess" ]; then
    LICHESS_FLAG="-lichess-mode"
    shift
fi

if [ $# -lt 2 ]; then
    echo "Usage: $0 [--lichess] <input.pgn[.zst]> <output_dir>"
    echo ""
    echo "Converts PGN (optionally zstd-compressed) to V6 training data."
    echo ""
    echo "Options:"
    echo "  --lichess  Extract Stockfish evals from Lichess game comments."
    echo "             Without this flag, value comes from game result only"
    echo "             (recommended for human imitation training)."
    exit 1
fi

INPUT="$(realpath "$1")"
OUTPUT_DIR="$(realpath -m "$2")"

if [ ! -x "$TOOL" ]; then
    echo "Error: trainingdata-tool not found at $TOOL"; exit 1
fi

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

# Step 1: Decompress if needed
if [[ "$INPUT" == *.zst ]]; then
    echo "=== Decompressing $(basename "$INPUT") ==="
    PGN_FILE="${TMPDIR}/games.pgn"
    zstd -d "$INPUT" -o "$PGN_FILE" --no-progress
    echo "  Decompressed to $(du -h "$PGN_FILE" | cut -f1)"
else
    PGN_FILE="$INPUT"
fi

# Step 2: Convert PGN to V4 training data
echo "=== Converting PGN to V4 training data ==="
if [ -n "$LICHESS_FLAG" ]; then
    echo "  (Lichess mode: extracting SF evals from comments)"
else
    echo "  (Standard mode: value from game result only)"
fi
V4_DIR="${TMPDIR}/v4_data"
mkdir -p "$V4_DIR"
pushd "$V4_DIR" > /dev/null
"$TOOL" $LICHESS_FLAG "$PGN_FILE"
popd > /dev/null
N_V4=$(find "$V4_DIR" -name '*.gz' | wc -l)
echo "  Generated $N_V4 V4 chunk files"

# Step 3: Convert V4 to V6
echo "=== Converting V4 to V6 training data ==="
mkdir -p "$OUTPUT_DIR"

for subdir in "$V4_DIR"/supervised-*; do
    if [ -d "$subdir" ]; then
        dirname=$(basename "$subdir")
        python3 "$V4_TO_V6" "$subdir" "${OUTPUT_DIR}/${dirname}"
    fi
done

N_V6=$(find "$OUTPUT_DIR" -name '*.gz' | wc -l)
echo ""
echo "=== Done ==="
echo "Output: $OUTPUT_DIR ($N_V6 V6 chunk files)"
