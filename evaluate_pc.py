from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch

from generate_pc_datasets import DEFAULT_PARAMS as PC_DATASET_PARAMS
from rl4co.data.utils import load_npz_to_tensordict, save_tensordict_to_npz
from rl4co.envs.pc.env import PartConsolidationEnv
from rl4co.envs.pc.evaluator import DEFAULT_MODULARITY_GAMMA, evaluate_groups, score_metric_rows
from rl4co.models.zoo.am import AttentionModel
from rl4co.utils.ops import unbatchify


DEFAULT_DATA = Path("data/pc/pc20_newdist_test_seed1234.npz")  # 1234
DEFAULT_CKPT_ROOT = Path("logs/train/curriculum")
PAPER_ALGORITHMS = ("ortools", "cpccd", "mcl", "nco")
PAPER_RANGES = ("10-20", "21-30", "31-40", "41-50")
PAPER_GAMMAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PC baselines and trained NCO checkpoints."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--algorithms",
        type=str,
        help="Comma-separated list from: ortools, cpccd, mcl, nco.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Number of instances to evaluate. Use 0 to evaluate all instances.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, default=Path("baseline_results.csv"))
    parser.add_argument("--gamma", type=float, default=DEFAULT_MODULARITY_GAMMA)
    parser.add_argument("--ortools-time-limit", type=float, default=300.0)
    parser.add_argument("--ortools-workers", type=int, default=1)
    parser.add_argument("--paper-run-root", type=Path, help="Run the paper benchmark using this curriculum run.")
    parser.add_argument("--paper-output-dir", type=Path, default=Path("outputs/paper_evaluation"))
    parser.add_argument("--ranges", nargs="+", choices=PAPER_RANGES, default=PAPER_RANGES)
    parser.add_argument("--gammas", nargs="+", type=float, choices=PAPER_GAMMAS, default=PAPER_GAMMAS)
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Path to an RL4CO checkpoint for the nco algorithm.",
    )
    parser.add_argument(
        "--auto-ckpt",
        action="store_true",
        help="Automatically use the latest best Stage 3 checkpoint under --ckpt-root.",
    )
    parser.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Device for NCO evaluation.",
    )
    parser.add_argument("--nco-batch-size", type=int, help="Defaults to 100 in paper mode, 256 otherwise.")

    parser.add_argument("--mcl-expansion", type=int, default=2)
    parser.add_argument("--mcl-inflation", type=float, default=2.0)
    parser.add_argument("--mcl-loop-value", type=float, default=1.0)
    parser.add_argument("--mcl-iterations", type=int, default=100)
    parser.add_argument("--mcl-pruning-threshold", type=float, default=0.001)
    parser.add_argument(
        "--cpccd-alpha",
        type=float,
        default=None,
        help=(
            "Enable the repository-specific strength-only fallback at this "
            "modularity threshold. Omit it for threshold-free CPCCD."
        ),
    )
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def find_latest_checkpoint(root: Path) -> Path:
    candidates = sorted(
        root.glob("**/stage3_n10-40/checkpoints/epoch_*.ckpt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No best Stage 3 checkpoint found under {root}. Use --ckpt to specify a checkpoint.")
    return candidates[0]


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def scalar(value: Any) -> Any:
    arr = to_numpy(value)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return arr


def strip_sep_instance(td_item: Any) -> dict[str, Any]:
    """Convert an RL4CO PC instance to the no-SEP format expected by baselines."""
    num_parts = int(scalar(td_item["num_parts"]))
    node_slice = slice(1, num_parts + 1)

    inst: dict[str, Any] = {
        "num_parts": num_parts,
    }

    if "material_type_count" in td_item.keys():
        inst["material_type_count"] = int(scalar(td_item["material_type_count"]))

    for key in (
        "node_features",
        "valid_part_mask",
        "material",
        "maintfreq",
        "isstandard",
        "pos1d",
    ):
        if key in td_item.keys():
            inst[key] = to_numpy(td_item[key])[node_slice]

    for key in (
        "edge_features",
        "W",
        "assembly_adj",
        "mat_var",
        "maint_diff",
        "rel_motion",
        "relation_valid",
        "relation_consistent",
    ):
        if key in td_item.keys():
            value = to_numpy(td_item[key])
            if value.ndim >= 2:
                inst[key] = value[node_slice, node_slice]
            else:
                inst[key] = clean_value(value)

    if "assembly_adj" not in inst and "W" in inst:
        inst["assembly_adj"] = np.asarray(inst["W"], dtype=float) > 1e-8
    if "relation_valid" not in inst and "W" in inst:
        inst["relation_valid"] = np.asarray(inst["W"], dtype=float) > 1e-8

    return inst


def make_solver(name: str, args: argparse.Namespace, seed: int) -> Any:
    name = name.lower()
    if name == "ortools":
        from baseline.ortools_pc_solver import ORToolsPCSolver

        return ORToolsPCSolver(
            gamma=args.gamma,
            time_limit_sec=args.ortools_time_limit,
            num_workers=args.ortools_workers,
        )
    if name == "cpccd":
        from baseline.cpccd_solver import CPCCDSolver

        return CPCCDSolver(alpha=args.cpccd_alpha)
    if name == "mcl":
        from baseline.mcl_solver import MCLSolver

        return MCLSolver(
            expansion=getattr(args, "mcl_expansion", 2),
            inflation=getattr(args, "mcl_inflation", 2.0),
            loop_value=getattr(args, "mcl_loop_value", 1.0),
            iterations=getattr(args, "mcl_iterations", 100),
            pruning_threshold=getattr(args, "mcl_pruning_threshold", 0.001),
        )
    raise ValueError(f"Unknown algorithm: {name}")


def make_env_from_dataset(dataset: Any, device: torch.device, gamma: float) -> PartConsolidationEnv:
    sample = dataset[0]
    generator_params = {
        "num_parts": dataset["valid_part_mask"].shape[-1] - 1,
        "material_types": int(scalar(sample["material_type_count"]))
        if "material_type_count" in sample.keys()
        else 3,
    }
    return PartConsolidationEnv(
        generator_params=generator_params, device=str(device), modularity_gamma=gamma
    )


def load_nco_model(ckpt_path: Path, env: PartConsolidationEnv, device: torch.device):
    # Load checkpoints on CPU first. Lightning checkpoints may contain serialized
    # env objects, and their RNG state must remain a CPU ByteTensor during unpickle.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    hparams = ckpt.get("hyper_parameters", {})
    policy_kwargs = dict(hparams.get("policy_kwargs", {}))
    metrics = hparams.get("metrics", {})
    if any(key.startswith("critic.") for key in state_dict) or (
        "num_starts" in hparams and "num_augment" in hparams
    ):
        raise ValueError("Expected an Attention Model REINFORCE checkpoint.")

    model = AttentionModel(
        env=env,
        policy_kwargs=policy_kwargs,
        baseline=hparams.get("baseline", "no"),
        metrics=metrics,
    )
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        print(f"NCO checkpoint missing keys: {len(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        print(f"NCO checkpoint unexpected keys: {len(incompatible.unexpected_keys)}")
    model.to(device)
    model.eval()
    return model, "am"


def clean_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def evaluate_algorithm(
    name: str,
    dataset: Any,
    limit: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for idx in range(limit):
        print(
            f"  Running {name}: instance {idx + 1}/{limit} "
            f"(dataset index {idx})...",
            flush=True,
        )
        seed = args.seed + idx
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        inst = strip_sep_instance(dataset[idx])
        solver = make_solver(name, args, seed)

        started = time.perf_counter()
        if name == "ortools":
            result = solver.solve_with_status(inst)
            groups, solver_elapsed = result.groups, result.elapsed_sec
        else:
            groups, solver_elapsed = solver.solve(inst)
        wall_elapsed = time.perf_counter() - started

        metrics = score_metric_rows([evaluate_groups(groups, inst, gamma=args.gamma)])[0]
        score = metrics["score"]

        row = {
            "algorithm": name,
            "gamma": args.gamma,
            "instance_idx": idx,
            "score": float(score),
            "reward": float(score),
            "solver_elapsed_sec": float(solver_elapsed),
            "wall_elapsed_sec": float(wall_elapsed),
            "groups": json.dumps(groups),
        }
        if name == "ortools":
            row.update(
                solver_status=result.status,
                solver_has_solution=result.status in {"OPTIMAL", "FEASIBLE", "ZERO_WEIGHT"},
                ortools_objective_value=result.objective_value,
                ortools_best_objective_bound=result.best_objective_bound,
            )
        if name.lower() == "mcl":
            row.update(
                {
                    "solver_status": (
                        "CONVERGED"
                        if solver.last_diagnostics.converged
                        else "MAX_ITERATIONS"
                    ),
                    "mcl_expansion": solver.expansion,
                    "mcl_inflation": solver.inflation,
                    "mcl_loop_value": solver.loop_value,
                    "mcl_iterations_limit": solver.iterations,
                    "mcl_iterations_used": solver.last_diagnostics.iterations,
                    "mcl_pruning_threshold": solver.pruning_threshold,
                    "mcl_implementation": solver.implementation,
                    "mcl_raw_clusters": json.dumps(solver.last_raw_clusters),
                    "mcl_raw_num_clusters": len(solver.last_raw_clusters),
                    "mcl_repair_splits": solver.last_repair_splits,
                    "mcl_weight_source": "W",
                }
            )
        row.update({key: clean_value(value) for key, value in metrics.items()})
        rows.append(row)

    return rows


def node_groups_to_part_groups(groups: list[list[int]]) -> list[list[int]]:
    return [[int(node) - 1 for node in group if int(node) > 0] for group in groups]


def select_nco_actions(out: dict[str, torch.Tensor], num_starts: int) -> tuple[torch.Tensor, torch.Tensor]:
    actions = out["actions"]
    reward = out["reward"]
    if num_starts and num_starts > 1:
        reward_by_start = unbatchify(reward, num_starts)
        actions_by_start = unbatchify(actions, num_starts)
        best_idx = reward_by_start.argmax(dim=1)
        batch_idx = torch.arange(actions_by_start.size(0), device=actions_by_start.device)
        actions = actions_by_start[batch_idx, best_idx]
        reward = reward_by_start[batch_idx, best_idx]
    return actions.detach().cpu(), reward.detach().cpu()


def evaluate_nco(
    dataset: Any,
    limit: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    ckpt_path = args.ckpt
    if ckpt_path is None:
        if not args.auto_ckpt:
            print("NCO checkpoint not provided; using latest checkpoint automatically.")
        ckpt_path = find_latest_checkpoint(args.ckpt_root)
    ckpt_path = ckpt_path.resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"NCO checkpoint not found: {ckpt_path}")

    device = resolve_device(args.device)
    env = make_env_from_dataset(dataset, device, args.gamma)
    model, model_kind = load_nco_model(ckpt_path, env, device)
    num_starts = int(getattr(model, "num_starts", 0) or 0)

    print(f"NCO checkpoint: {ckpt_path}")
    print(f"NCO model kind: {model_kind}")
    print(f"NCO device: {device}")

    rows: list[dict[str, Any]] = []
    for start in range(0, limit, args.nco_batch_size):
        end = min(start + args.nco_batch_size, limit)
        batch = dataset[start:end].to(device)
        td = env.reset(batch)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.no_grad():
            if num_starts > 1:
                out = model.policy(td.clone(), env, phase="test", num_starts=num_starts)
            else:
                out = model.policy(td.clone(), env, phase="test")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        wall_elapsed = time.perf_counter() - started

        actions, model_reward = select_nco_actions(out, num_starts)
        td_cpu = td.cpu()
        node_groups = env.actions_to_groups(actions, td_cpu)
        per_instance_elapsed = wall_elapsed / max(end - start, 1)

        for local_idx, groups_with_sep in enumerate(node_groups):
            idx = start + local_idx
            inst = strip_sep_instance(dataset[idx])
            groups = node_groups_to_part_groups(groups_with_sep)
            metrics = score_metric_rows([evaluate_groups(groups, inst, gamma=args.gamma)])[0]
            score = metrics["score"]
            row = {
                "algorithm": "nco",
                "gamma": args.gamma,
                "instance_idx": idx,
                "score": float(score),
                "reward": float(score),
                "model_reward": float(model_reward[local_idx].item()),
                "solver_elapsed_sec": float(per_instance_elapsed),
                "wall_elapsed_sec": float(per_instance_elapsed),
                "groups": json.dumps(groups),
                "ckpt": str(ckpt_path),
                "model_kind": model_kind,
            }
            row.update({key: clean_value(value) for key, value in metrics.items()})
            rows.append(row)

    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    preferred = [
        "algorithm",
        "instance_idx",
        "score",
        "reward",
        "feasible",
        "num_groups",
        "Q_observed",
        "Q_expected",
        "solver_elapsed_sec",
        "wall_elapsed_sec",
        "groups",
    ]
    keys = {key for row in rows for key in row}
    fieldnames = [key for key in preferred if key in keys] + sorted(keys - set(preferred))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["algorithm"]].append(row)

    print("\nSummary")
    print("-" * 80)
    for name, alg_rows in grouped.items():
        score = np.mean([row["score"] for row in alg_rows])
        feasible = np.mean([float(row.get("feasible", 0.0)) for row in alg_rows])
        num_groups = np.mean([float(row.get("num_groups", 0.0)) for row in alg_rows])
        elapsed = np.mean([row["wall_elapsed_sec"] for row in alg_rows])
        total_elapsed = np.sum([row["wall_elapsed_sec"] for row in alg_rows])
        print(
            f"{name:>6} | n={len(alg_rows):4d} | score={score: .6f} | "
            f"feasible={feasible: .3f} | groups={num_groups: .2f} | "
            f"time={elapsed: .4f}s/inst | total={total_elapsed: .4f}s"
        )


def evaluate_algorithms(dataset, limit, args, algorithms):
    rows = []
    for name in algorithms:
        print(f"\nRunning {name}...")
        if name == "nco":
            rows.extend(evaluate_nco(dataset, limit, args))
        else:
            rows.extend(evaluate_algorithm(name, dataset, limit, args))
    return rows


def load_paper_dataset(test_range: str, seed: int):
    low, high = map(int, test_range.split("-"))
    path = Path("data/pc") / f"pc{test_range}_newdist_test100_seed{seed}.npz"
    if not path.exists():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            env = PartConsolidationEnv(
                generator_params={**PC_DATASET_PARAMS, "num_parts": low, "max_num_parts": high}
            )
            dataset = env.generator(100)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_tensordict_to_npz(dataset, path)
        print(f"Generated fixed test dataset: {path}")
    dataset = load_npz_to_tensordict(path)
    sizes = dataset["num_parts"]
    if dataset.batch_size[0] != 100 or not ((sizes >= low) & (sizes <= high)).all():
        raise ValueError(f"Expected 100 instances in the range {test_range}: {path}")
    return dataset


def summarize_condition(rows, condition):
    import pandas as pd
    from scipy.stats import friedmanchisquare, wilcoxon

    frame = pd.DataFrame(rows)
    summary = frame.groupby("algorithm", sort=False).agg(
        n=("score", "size"),
        score_mean=("score", "mean"),
        score_std=("score", "std"),
        num_groups_mean=("num_groups", "mean"),
        feasible_rate=("feasible", "mean"),
        wall_elapsed_sec_mean=("wall_elapsed_sec", "mean"),
    )
    summary_rows = [
        {**condition, "algorithm": algorithm, **metrics}
        for algorithm, metrics in summary.to_dict("index").items()
    ]
    paired = frame.pivot(index="instance_idx", columns="algorithm", values="score")
    if len(paired) < 2 or not set(PAPER_ALGORITHMS).issubset(paired.columns):
        return summary_rows, [], []
    scores = paired[list(PAPER_ALGORITHMS)].to_numpy()
    if np.all(scores == scores[:, :1]):
        omnibus_statistic, omnibus_p = 0.0, 1.0
    else:
        omnibus_statistic, omnibus_p = friedmanchisquare(*scores.T)
    friedman_rows = [{
        **condition,
        "n": len(paired),
        "friedman_statistic": omnibus_statistic,
        "friedman_p": omnibus_p,
    }]
    pairwise = []
    for a, b in combinations(PAPER_ALGORITHMS, 2):
        differences = (paired[a] - paired[b]).to_numpy(copy=True)
        differences[np.isclose(differences, 0.0, atol=1e-10)] = 0.0
        statistic, p = wilcoxon(differences) if differences.any() else (0.0, 1.0)
        pairwise.append({
            **condition,
            "algorithm_a": a,
            "algorithm_b": b,
            "n": len(paired),
            "mean_difference_a_minus_b": float(differences.mean()),
            "wilcoxon_statistic": statistic,
            "wilcoxon_p": p,
        })
    adjusted = 0.0
    for i, row in enumerate(sorted(pairwise, key=lambda row: row["wilcoxon_p"])):
        adjusted = max(adjusted, min(1.0, (len(pairwise) - i) * row["wilcoxon_p"]))
        row["wilcoxon_p_holm"] = adjusted
        row["significant_at_0_05"] = bool(omnibus_p < 0.05 and adjusted < 0.05)
    return summary_rows, friedman_rows, pairwise


def run_paper_benchmark(args, algorithms):
    checkpoints = {}
    if "nco" in algorithms:
        for gamma in args.gammas:
            label = f"{gamma:g}".replace(".", "p")
            stage = args.paper_run_root / f"gamma_{label}" / "stage3_n10-40" / "checkpoints"
            candidates = sorted(stage.glob("epoch_*.ckpt"), key=lambda p: p.stat().st_mtime)
            if not candidates:
                raise FileNotFoundError(f"No best Stage 3 checkpoint found in {stage}")
            checkpoints[gamma] = candidates[-1]

    output = args.paper_output_dir
    output.mkdir(parents=True, exist_ok=False)
    raw, summaries, friedman, pairwise = [], [], [], []
    for test_range in dict.fromkeys(args.ranges):
        dataset = load_paper_dataset(test_range, args.seed)
        limit = min(args.limit, len(dataset)) if args.limit > 0 else len(dataset)
        for gamma in dict.fromkeys(args.gammas):
            print(f"\nTest range {test_range}, gamma={gamma:g}, instances={limit}", flush=True)
            condition = {"test_range": test_range, "gamma": gamma}
            condition_args = argparse.Namespace(**vars(args))
            condition_args.gamma = gamma
            condition_args.ckpt = checkpoints.get(gamma)
            rows = evaluate_algorithms(dataset, limit, condition_args, algorithms)
            raw.extend({**row, **condition} for row in rows)
            summary_rows, friedman_rows, pairwise_rows = summarize_condition(rows, condition)
            summaries.extend(summary_rows)
            friedman.extend(friedman_rows)
            pairwise.extend(pairwise_rows)
            write_csv(output / "raw_results.csv", raw)
            write_csv(output / "summary.csv", summaries)
            write_csv(output / "friedman_tests.csv", friedman)
            write_csv(output / "pairwise_wilcoxon_holm.csv", pairwise)
    print(f"\nSaved paper benchmark results to: {output}")


def main() -> None:
    args = parse_args()
    if args.nco_batch_size is None:
        args.nco_batch_size = 100 if args.paper_run_root else 256
    defaults = PAPER_ALGORITHMS if args.paper_run_root else ("cpccd", "mcl")
    algorithms = (
        list(dict.fromkeys(item.strip().lower() for item in args.algorithms.split(",") if item.strip()))
        if args.algorithms is not None else list(defaults)
    )
    if not algorithms or set(algorithms) - set(PAPER_ALGORITHMS):
        raise ValueError(f"Select algorithms from: {', '.join(PAPER_ALGORITHMS)}")
    if args.nco_batch_size < 1 or args.ortools_workers < 1 or args.ortools_time_limit <= 0:
        raise ValueError("Batch size, worker count, and time limit must be positive.")
    if args.paper_run_root:
        run_paper_benchmark(args, algorithms)
        return

    dataset = load_npz_to_tensordict(args.data)
    dataset_size = int(dataset.batch_size[0])
    limit = dataset_size if args.limit <= 0 else min(args.limit, dataset_size)

    print(f"Data: {args.data}")
    print(f"Instances: {limit}/{dataset_size}")
    print(f"Algorithms: {', '.join(algorithms)}")

    all_rows = evaluate_algorithms(dataset, limit, args, algorithms)
    write_csv(args.output, all_rows)
    print_summary(all_rows)
    print(f"\nSaved results to: {args.output}")


if __name__ == "__main__":
    main()
