#!/usr/bin/env python3
"""Convert V4TrainingData .gz files to V6TrainingData format.

Supports parallel conversion via --workers (defaults to CPU count).
"""

import argparse
import gzip
import os
import struct
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

V4_SIZE = 8292
V6_SIZE = 8356

V4_FMT = "<I 1858f 104Q 4B 3B b 4f"
V6_FMT = "<I I 1858f 104Q 4B B B B B 4f 3f 2f 3f 3f I 2H 2f"

NAN = float("nan")

# Pre-compile struct objects for speed.
_v4_struct = struct.Struct(V4_FMT)
_v6_struct = struct.Struct(V6_FMT)


def convert_v4_to_v6(v4_bytes):
    fields = _v4_struct.unpack(v4_bytes)
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
    return _v6_struct.pack(*v6_values)


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


def _worker(args):
    """Worker function for multiprocessing Pool."""
    input_path, output_path = args
    try:
        return convert_gz_file(input_path, output_path)
    except Exception as e:
        print(f"Error converting {input_path}: {e}", file=sys.stderr)
        return 0


def main():
    parser = argparse.ArgumentParser(description="Convert V4 .gz to V6 format")
    parser.add_argument("input", help="Input dir with V4 .gz files")
    parser.add_argument("output", help="Output dir for V6 .gz files")
    parser.add_argument(
        "-j", "--workers", type=int, default=0,
        help="Number of parallel workers (default: CPU count)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    gz_files = sorted(input_dir.rglob("*.gz"))
    if not gz_files:
        print(f"No .gz files found in {input_dir}")
        sys.exit(1)

    num_workers = args.workers if args.workers > 0 else cpu_count()

    # Build list of (input, output) pairs, skipping already-converted files.
    work_items = []
    skipped = 0
    for gz_file in gz_files:
        rel_path = gz_file.relative_to(input_dir)
        out_path = output_dir / rel_path
        if out_path.exists():
            skipped += 1
        else:
            work_items.append((str(gz_file), str(out_path)))

    total = len(gz_files)
    to_convert = len(work_items)
    print(f"Found {total} .gz files: {to_convert} to convert, {skipped} already done")

    if to_convert == 0:
        print("Nothing to do.")
        return

    if num_workers == 1:
        # Sequential mode
        total_frames = 0
        for i, (inp, outp) in enumerate(work_items):
            total_frames += _worker((inp, outp))
            if (i + 1) % 100 == 0 or (i + 1) == to_convert:
                print(f"  [{i+1}/{to_convert}] {total_frames} frames")
    else:
        # Parallel mode
        print(f"Converting with {num_workers} workers...")
        total_frames = 0
        done = 0
        t0 = time.perf_counter()
        with Pool(num_workers) as pool:
            for n in pool.imap_unordered(_worker, work_items, chunksize=8):
                total_frames += n
                done += 1
                if done % 200 == 0 or done == to_convert:
                    elapsed = time.perf_counter() - t0
                    rate = done / elapsed
                    eta = (to_convert - done) / rate if rate > 0 else 0
                    print(
                        f"  [{done}/{to_convert}] {total_frames} frames, "
                        f"{rate:.1f} files/s, ETA {eta/60:.0f}m"
                    )

    print(f"Done. {to_convert} files converted, {total_frames} frames total.")


if __name__ == "__main__":
    main()
