#!/usr/bin/env python3
"""Convert V4TrainingData .gz files to V6TrainingData format."""

import argparse
import gzip
import math
import os
import struct
import sys
from pathlib import Path

V4_SIZE = 8292
V6_SIZE = 8356

V4_FMT = "<I 1858f 104Q 4B 3B b 4f"
V6_FMT = "<I I 1858f 104Q 4B B B B B 4f 3f 2f 3f 3f I 2H 2f"

NAN = float("nan")


def convert_v4_to_v6(v4_bytes):
    fields = struct.unpack(V4_FMT, v4_bytes)
    idx = 0
    version = fields[idx]; idx += 1
    probs = fields[idx:idx+1858]; idx += 1858
    planes = fields[idx:idx+104]; idx += 104
    c_us_ooo = fields[idx]; idx += 1
    c_us_oo = fields[idx]; idx += 1
    c_them_ooo = fields[idx]; idx += 1
    c_them_oo = fields[idx]; idx += 1
    side_to_move = fields[idx]; idx += 1
    rule50_count = fields[idx]; idx += 1
    move_count = fields[idx]; idx += 1
    result = fields[idx]; idx += 1
    root_q = fields[idx]; idx += 1
    best_q = fields[idx]; idx += 1
    root_d = fields[idx]; idx += 1
    best_d = fields[idx]; idx += 1

    # Derive result WDL from game outcome.
    if result == 1:
        result_q, result_d = 1.0, 0.0
    elif result == -1:
        result_q, result_d = -1.0, 0.0
    else:
        result_q, result_d = 0.0, 1.0

    # Use the game result as the value target everywhere.
    # This ensures consistency regardless of which value_type
    # the loss function is configured to use.
    # If the V4 data has engine evals (lichess mode), prefer those
    # for root_q/best_q; otherwise use the game result.
    has_eval = (root_q != 0.0 or best_q != 0.0)
    if not has_eval:
        root_q = result_q
        best_q = result_q
        root_d = result_d
        best_d = result_d

    # Find the played move index (highest probability).
    max_prob = max(probs)
    played_idx = probs.index(max_prob) if max_prob > 0 else 0

    v6_values = (
        6, 1,                                        # version, input_format
        *probs, *planes,                             # probabilities, planes
        c_us_ooo, c_us_oo, c_them_ooo, c_them_oo,   # castling
        side_to_move, rule50_count, 0, 0,            # stm, r50, inv, dummy
        root_q, best_q, root_d, best_d,              # root/best q/d
        0.0, 0.0,                                    # root_m, best_m
        0.0,                                         # plies_left
        result_q, result_d,                          # result
        root_q, root_d,                              # played_q, played_d
        0.0,                                         # played_m
        NAN, NAN, NAN,                               # orig_q/d/m
        0,                                           # visits
        played_idx, played_idx,                      # played_idx, best_idx
        0.0, 0.0,                                    # policy_kld, q_st
    )
    return struct.pack(V6_FMT, *v6_values)


def convert_gz_file(input_path, output_path):
    with gzip.open(input_path, "rb") as f:
        data = f.read()
    if len(data) % V4_SIZE != 0:
        print(f"Warning: {input_path} size not multiple of V4, skipping", file=sys.stderr)
        return 0
    num_frames = len(data) // V4_SIZE
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with gzip.open(output_path, "wb") as f:
        for i in range(num_frames):
            f.write(convert_v4_to_v6(data[i*V4_SIZE:(i+1)*V4_SIZE]))
    return num_frames


def main():
    parser = argparse.ArgumentParser(description="Convert V4 .gz to V6 format")
    parser.add_argument("input", help="Input dir with V4 .gz files")
    parser.add_argument("output", help="Output dir for V6 .gz files")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    gz_files = sorted(input_dir.rglob("*.gz"))
    if not gz_files:
        print(f"No .gz files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(gz_files)} .gz files to convert")
    total_frames = 0
    for i, gz_file in enumerate(gz_files):
        rel_path = gz_file.relative_to(input_dir)
        out_path = output_dir / rel_path
        n = convert_gz_file(str(gz_file), str(out_path))
        total_frames += n
        if (i + 1) % 100 == 0 or (i + 1) == len(gz_files):
            print(f"  [{i+1}/{len(gz_files)}] {total_frames} frames")

    print(f"Done. {len(gz_files)} files, {total_frames} frames.")


if __name__ == "__main__":
    main()
