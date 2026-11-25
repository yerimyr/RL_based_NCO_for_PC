# Reinforcement Learning-Based Neural Combinatorial Optimization for Part Consolidation under Manufacturing Constraints

Research code by Yerim Ji

This repository contains the training code and baseline implementations for the paper. Part consolidation groups components of an existing assembly into manufacturable parts. The objective rewards retaining strong assembly relationships within groups, with a resolution parameter γ that adjusts the modularity penalty. Each group must be connected and satisfy the material, maintenance, standard-part, and relative-motion constraints.

The proposed policy uses an edge-aware graph encoder and an attention decoder to construct a feasible partition one action at a time. It is trained with REINFORCE and a greedy rollout baseline. The benchmark compares it with CP-SAT, Community-based Part Consolidation Candidate Detection (CPCCD), and Markov Clustering (MCL) on the same instances.

## Setup

Run commands from the repository root in PowerShell. Use Python 3.10 or later in an isolated environment. Set up a CUDA-enabled PyTorch installation for GPU training.

```powershell
python -m pip install -e .
python -m pip install pandas networkx ortools==9.15.6755 tensorboard
```

On a fresh clone, generate the fixed validation and test datasets before training:

```powershell
python generate_pc_datasets.py --num-parts 20 30 40 --material-types 2 --val-seed 4321 --test-seed 1234
```

This writes the files used by the three curriculum stages to `data/pc/`.

## Train the policies

From the repository root, run:

```powershell
python run.py experiment=pc/am_pc_edge hydra.run.dir=logs/train/curriculum/paper_reproduction
```

The paper experiment trains all five γ values (`0, 0.25, 0.5, 0.75, 1`) with three curriculum stages per value. Omit `hydra.run.dir` to use a timestamped directory under `logs/train/curriculum/`. Use a fresh directory for a new run.

| Stage | 10–20 parts | 21–30 parts | 31–40 parts |
| --- | ---: | ---: | ---: |
| 1 | 100% | 0% | 0% |
| 2 | 20% | 80% | 0% |
| 3 | 10% | 10% | 80% |

Each stage uses 100 epochs, 10,000 training instances per epoch, batch size 128, seed 1234, and at most two material types. Stages 2 and 3 automatically load the previous stage's best policy weights and start a fresh optimizer. The learning rate is `1e-4`, with `1e-5` in stages 2 and 3 for γ = 0.5, 0.75, and 1.

For a single training job with manual overrides, add `curriculum.enabled=false`.

To evaluate a trained policy on the fixed 20-part test set, run:

```powershell
python evaluate_pc.py --algorithms nco --auto-ckpt --device cpu
```

`--auto-ckpt` picks the most recently modified best Stage 3 checkpoint. Use `--ckpt` to evaluate a specific run or γ.

## Repository Structure

```text
.
├── run.py                              # Curriculum training entry point
├── generate_pc_datasets.py             # Fixed validation and test data generation
├── evaluate_pc.py                      # CPCCD, MCL, and trained NCO evaluation
├── configs/
│   ├── env/pc.yaml                     # PC environment and data settings
│   └── experiment/pc/am_pc_edge.yaml   # Paper's training configuration
├── baseline/
│   ├── cpccd_solver.py                 # CPCCD baseline
│   ├── mcl_solver.py                   # MCL baseline
│   ├── ortools_pc_solver.py            # CP-SAT baseline
│   └── third_party/markov_clustering/  # Vendored MCL code and license
├── rl4co/
│   ├── envs/pc/                        # PC environment and instance generator
│   ├── models/nn/graph/pc_encoder.py   # Edge-aware graph encoder
│   └── tasks/train.py                  # Curriculum orchestration
├── tests/                              # Environment, model, and solver tests
└── data/pc/                            # Generated validation and test datasets
```

`data/pc/` is generated locally. Training creates `logs/train/curriculum/` for checkpoints and TensorBoard logs.

## Citation and licensing

See [CITATION.cff](CITATION.cff) for the software citation. This implementation builds on RL4CO. Its copyright and MIT license are retained in [LICENSE](LICENSE). Additional third-party notices are in [baseline/third_party_notices.md](baseline/third_party_notices.md).
