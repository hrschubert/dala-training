# Dala — Human-Aligned Chess Networks

Dala is a series of chess neural networks aligned to specific Elo rating brackets, designed to play at a target skill level rather than at maximum strength. The networks are trained for the following rating brackets:
**700, 900, 1100, 1300 1600**


This repository is a fork of [LeelaChessZero / lczero-training](https://github.com/LeelaChessZero/lczero-training), modified so that **BT4+ transformer networks** can be trained from converted PGN files. The training data used in this project consists of [Lichess](https://lichess.org/) standard rated games from **January 2024 through February 2026**, filtered by Elo bracket.

## Network Releases & Lichess Bots

The trained networks are published under the [Releases](../../releases) page of this repository. They are also deployed as Lichess bots that anyone can challenge:

| Targeted Rating | Lichess Bot |
|---|---|
| 700  | [@dala-700](https://lichess.org/@/dala-700)   |
| 900  | [@dala-900](https://lichess.org/@/dala-900)   |
| 1100 | [@dala-1100](https://lichess.org/@/dala-1100) |
| 1300 | [@dala-1300](https://lichess.org/@/dala-1300) |
| 1600 | [@dala-1600](https://lichess.org/@/dala-1600) |

### Move Selection: Weighted Random vs. Best Move

Unlike the [Maia](https://maiachess.com/) project, which deploys its networks using a `best_move` (argmax) policy, the Dala bots play **weighted random moves at depth 1**. The move at each turn is sampled from the policy distribution produced by the network rather than always selecting the most-probable move.

This is a deliberate alignment choice:

- **`best_move` selection inflates strength** — even a network trained on 1100-rated games will play noticeably stronger than 1100 if it always picks the most-probable move, because it filters out the human noise that defines play at that level.
- **Weighted random sampling preserves the policy distribution**, including the human-typical mistakes, so the bot's actual playing strength tracks the targeted Elo bracket.

The modified Lichess bot client used to deploy these networks is available at [hrschubert/lichess-bot](https://github.com/hrschubert/lichess-bot).

## Statistics
| Targeted Rating | Move accuracy (top-5) | Policy loss
|---|---|---|
| 1100 | 89.36% | 1.4635
| 1300 | 90.04% | 1.4257
| 1600 | 91.70% | 1.3387

Table reports the top-5 move accuracy and policy loss across the highest targeted Elo brackets. The results indicate that, for these rating levels, the model’s selected move appears among the five highest-scoring candidate moves in approximately 90% of positions. Accuracy increases with targeted Elo, rising from 89.36% at 1100 Elo to 91.70% at 1600 Elo, while policy loss decreases correspondingly from 1.4635 to 1.3387. This pattern suggests that play in higher Elo brackets is more predictable under the model’s policy distribution, yielding higher top-5 agreement and lower loss values.


## Changes vs. Upstream lczero-training

This fork extends the upstream training pipeline with the following:

### Training Pipeline
- **Initialize training from an exported network** — `lc0-train --from-network model.pb.gz` skips the separate `lc0-init` step and starts training directly from a published Leela network.
- **Periodic checkpointing & export** — `--checkpoint-every N` and `--export-every N` flags on `lc0-train` save intermediate checkpoints and `.pb.gz` exports during long runs (the upstream pipeline only saves at the end).
- **Per-step LR + loss logging** — the training log includes the current learning rate and a breakdown of the unweighted loss components on every step, simplifying loss/LR-divergence debugging.
- **Top-1 / top-5 move accuracy in `lc0-eval`** — the evaluation tool reports cross-entropy losses **and** policy accuracy on the entire dataset, with illegal-move masking applied.

### PGN → Training Data Pipeline
- `scripts/parallel_pgn_convert.sh` — parallelized conversion of large PGN files (50–200 GB Lichess monthly dumps) to V4 training data. Game-boundary splitting is done with `python-chess` to guarantee correctness, and chunk extraction runs across all available CPU cores.
- `tools/v4_to_v6.py` — multiprocessing-based V4 → V6 converter (≈6× faster than serial). Supports resumable runs.
- `tools/pgn_to_training_data.sh` — single-file pipeline driver: `.pgn.zst` → `zstd -d` → `trainingdata-tool` → V4 → V6.

### Tooling & Reproducibility
- `scripts/docker_setup.sh` — fully automated 10-step setup of the training environment in an Ubuntu + NVIDIA-driver Docker container (Miniconda, repo clone, C++ dataloader build, protobuf compilation, `trainingdata-tool` build, full PGN → V6 conversion, checkpoint init).
- `scripts/plot_training.py` — Matplotlib plotter that parses the training log and produces a PDF with smoothed loss curves and the learning-rate schedule on a dual y-axis.

### Supervised Training Restored
The upstream README notes that **"Generating trainingdata from pgn files is currently broken and has low priority"** — this fork restores and extends that path so the entire pipeline (raw Lichess PGN → trained Leela network) is reproducible end-to-end.

## Quick Start

### 1. Build the C++ dataloader and protobufs (Linux/WSL2)

```bash
git submodule update --init --recursive
uv venv
uv sync
CXX=clang++ CC=clang uv run meson setup build/release/ --buildtype=release --native-file=native.ini
just build-proto
meson compile -C build/release/
ln -s -T ../../build/release/_lczero_training.cpython-311-x86_64-linux-gnu.so \
   src/lczero_training/_lczero_training.so
```

(Or run `scripts/docker_setup.sh` inside an Ubuntu + NVIDIA Docker container to do all of the above plus PGN conversion automatically.)

### 2. Convert PGN to V6 training data

```bash
# Parallel PGN → V4
scripts/parallel_pgn_convert.sh filtered_1100.pgn /data/v4

# V4 → V6
python tools/v4_to_v6.py /data/v4 /data/v6 -j $(nproc)
```

### 3. Train

```bash
# Fresh training from a Leela network
lc0-train --config configs/dala_1100.textproto \
          --from-network base_network.pb.gz \
          --override-steps 0 \
          --checkpoint-every 5000 \
          --export-every 10000

# Resume from existing checkpoint
lc0-train --config configs/dala_1100.textproto --checkpoint-every 5000
```

### 4. Evaluate

```bash
lc0-eval --config configs/dala_1100.textproto --network dala_1100.pb.gz
```

### 5. Plot Training Progress

```bash
python scripts/plot_training.py training.log training.pdf
```

---

## Original README (excerpted)

The training pipeline resides in `src/lczero_training` and uses **JAX / Flax / Optax** (the legacy `tf/` TensorFlow pipeline is unmaintained).

### Installation

Install with `uv sync` (Python 3.11+). The C++ dataloader extension is built via `meson`. Configuration is via protobuf TextProto files; see `docs/example.textproto` for a complete reference.

### Data Preparation (upstream path)

The upstream pipeline expects pre-packaged self-play training data:

```
wget https://storage.lczero.org/files/training_data/training-run1--20200711-2017.tar
tar -xzf training-run1--20200711-2017.tar
```

The dataloader accepts only `.gz` or `.tar` files containing V6/V7 `TrainingData` binary records.

### Training Configuration

Configuration is via a `.textproto` file. Key sections control the model architecture (transformer blocks, `d_model`, SmolGen), the optimizer (NAdamW), the learning-rate schedule, and the data loader. See `docs/example.textproto` for the full schema.

### Building the 2025-08 release

1. Make sure `uv`, `just`, `meson` (and `protoc`) are installed.
2. `git submodule update --init --recursive`
3. `uv venv` *(do this **before** running meson, or it will build the module for the wrong Python)*
4. `uv sync`
5. `CXX=clang++ CC=clang uv run meson setup build/release/ --buildtype=release --native-file=native.ini`
6. `just build-proto`
7. `meson compile -C build/release/`
8. `ln -s -T ../../build/release/_lczero_training.cpython-311-x86_64-linux-gnu.so src/lczero_training/_lczero_training.so`
9. Run it: `uv run tui --config docs/example.textproto`

---

## License

This project inherits the GPL-3.0 license from upstream Leela Chess Zero. See `libs/lc0/LICENSE`.

## Acknowledgements

- The [LeelaChessZero](https://lczero.org/) project, on which this entire training pipeline is built.
- The [Maia Chess](https://maiachess.com/) project, whose Elo-bracketed approach to human-aligned chess engines inspired Dala.
- [Lichess](https://lichess.org/) for providing the [database of rated games](https://database.lichess.org/) that makes this work possible.
