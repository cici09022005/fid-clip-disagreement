import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


CONFIG_PATHS = {
    "anime_student": "kmc_lora/configs/base.yaml",
    "wikiart_mixed": "kmc_lora/configs/wikiart_mixed_scale_aware_local.yaml",
    "dreambooth_mixed": "kmc_lora/configs/dreambooth_mixed_scale_aware_local.yaml",
    "dreambooth_single": "kmc_lora/configs/dreambooth_single_scale_aware_local.yaml",
}

DEFAULT_MULTI_SEED_STRATEGIES = ["random:D-High", "kmc:D-High", "anti"]
DEFAULT_MULTI_SEEDS = [42, 52, 62, 72, 82]
DEFAULT_HPARAM_SEEDS = [42, 52, 62]
DEFAULT_RANKS = [8, 16, 32]
DEFAULT_STEPS = [600, 1200, 2400]
DEFAULT_PARETO_STRATEGY = {
    "anime_student": "anti",
    "wikiart_mixed": "anti",
    "dreambooth_mixed": "random:D-High",
    "dreambooth_single": "random:D-Sub-50",
}


def run(cmd, log_file=None, dry_run=False):
    print(" ".join(cmd), flush=True)
    if dry_run:
        return 0
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w", encoding="utf-8") as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        return result.returncode
    result = subprocess.run(cmd, text=True)
    return result.returncode


def load_cfg(dataset_name):
    config_path = CONFIG_PATHS[dataset_name]
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return config_path, cfg


def parse_strategy_spec(spec):
    if ":" in spec:
        strategy, split_name = spec.split(":", 1)
        strategy = strategy.strip().lower()
        split_name = split_name.strip()
    else:
        strategy = spec.strip().lower()
        split_name = None
    if strategy not in {"random", "kmc", "anti", "quality"}:
        raise ValueError(f"unknown strategy spec: {spec}")
    if strategy in {"random", "kmc"} and not split_name:
        raise ValueError(f"strategy requires split suffix: {spec}")
    return {"raw": spec, "strategy": strategy, "split": split_name}


def strategy_tag(spec_info):
    strategy = spec_info["strategy"]
    split_name = spec_info["split"]
    if strategy == "random":
        return f"Random_{split_name}"
    if strategy == "kmc":
        return f"KMC_{split_name}"
    if strategy == "anti":
        return "Anti_Curriculum"
    if strategy == "quality":
        return "Quality_Filter"
    raise ValueError(f"unsupported strategy: {strategy}")


def get_single_list_path(cfg, spec_info):
    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    strategy = spec_info["strategy"]
    split_name = spec_info["split"]
    if strategy == "random":
        return artifacts_dir / "splits" / f"{split_name}.txt"
    if strategy == "anti":
        return artifacts_dir / "lists" / "AntiCurriculum.txt"
    if strategy == "quality":
        return artifacts_dir / "lists" / "Quality-Top.txt"
    raise ValueError(f"single-list path not supported for strategy {strategy}")


def ensure_split_phase_lists(cfg, split_name):
    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    split_dir = artifacts_dir / "split_phases" / split_name
    required = [split_dir / "phase1.txt", split_dir / "phase2.txt", split_dir / "phase3.txt"]
    if all(path.exists() for path in required):
        return split_dir

    curriculum_csv = artifacts_dir / "curriculum.csv"
    split_file = artifacts_dir / "splits" / f"{split_name}.txt"
    if not curriculum_csv.exists():
        raise FileNotFoundError(f"missing curriculum file: {curriculum_csv}")
    if not split_file.exists():
        raise FileNotFoundError(f"missing split file: {split_file}")

    df = pd.read_csv(curriculum_csv)
    split_paths = {
        p.strip()
        for p in split_file.read_text(encoding="utf-8").splitlines()
        if p.strip()
    }
    split_df = df[df["path"].isin(split_paths)].copy()
    if split_df.empty:
        raise ValueError(f"no overlap between curriculum and split: {split_file}")

    split_dir.mkdir(parents=True, exist_ok=True)
    if all(col in split_df.columns for col in ["phase1", "phase2", "phase3"]):
        phase_to_paths = {
            phase_name: split_df.loc[split_df[phase_name].astype(bool), "path"]
            for phase_name in ["phase1", "phase2", "phase3"]
        }
    else:
        phase1_ratio = cfg["curriculum"]["phase1_ratio"]
        phase3_ratio = cfg["curriculum"]["phase3_ratio"]
        ordered = split_df.sort_values("difficulty", ascending=True).reset_index(drop=True)
        n = len(ordered)
        n1 = int(n * phase1_ratio)
        n3 = max(1, int(round(n * phase3_ratio)))
        phase_to_paths = {
            "phase1": ordered.loc[: max(0, n1 - 1), "path"] if n1 > 0 else ordered.iloc[0:0]["path"],
            "phase2": ordered["path"],
            "phase3": ordered.sort_values("typicality", ascending=False).head(n3)["path"],
        }

    for phase_name, phase_paths in phase_to_paths.items():
        phase_paths.to_csv(split_dir / f"{phase_name}.txt", index=False, header=False)
    return split_dir


def phase_step_plan(cfg, total_steps):
    ratios = cfg["curriculum"]
    p1 = max(1, int(round(total_steps * ratios["phase1_ratio"])))
    p2 = max(1, int(round(total_steps * ratios["phase2_ratio"])))
    p3 = max(1, total_steps - p1 - p2)
    return {"phase1": p1, "phase2": p2, "phase3": p3}


def train_single_phase(cfg, image_list, output_dir, seed, total_steps, rank, lr=None, dry_run=False):
    model_cfg = cfg["model"]
    prompts_cfg = cfg["prompts"]
    train_lr = lr if lr is not None else 1e-4
    cmd = [
        sys.executable,
        "kmc_lora/scripts/train_lora.py",
        "--base-model",
        str(model_cfg["base_model"]),
        "--image-list",
        str(image_list),
        "--output-dir",
        str(output_dir),
        "--instance-prompt",
        str(prompts_cfg["instance_prompt"]),
        "--resolution",
        str(model_cfg["resolution"]),
        "--train-batch-size",
        str(model_cfg["train_batch_size"]),
        "--gradient-accumulation-steps",
        str(model_cfg["gradient_accumulation_steps"]),
        "--max-train-steps",
        str(total_steps),
        "--save-steps",
        str(max(0, min(int(model_cfg["save_steps"]), int(total_steps)))),
        "--lr",
        str(train_lr),
        "--lora-rank",
        str(rank),
        "--lora-alpha",
        str(rank),
        "--seed",
        str(seed),
    ]
    validation_prompts = prompts_cfg.get("validation_prompts") or []
    if validation_prompts:
        cmd += ["--validation-prompts", *validation_prompts]
    return run(cmd, log_file=Path(output_dir) / "train.log", dry_run=dry_run)


def train_kmc(cfg, split_name, output_dir, seed, total_steps, rank, dry_run=False):
    model_cfg = cfg["model"]
    prompts_cfg = cfg["prompts"]
    split_phase_dir = ensure_split_phase_lists(cfg, split_name)
    step_plan = phase_step_plan(cfg, total_steps)
    phases = [
        ("phase1", split_phase_dir / "phase1.txt", 5e-5, None),
        ("phase2", split_phase_dir / "phase2.txt", 1e-4, output_dir / "phase1" / "final"),
        ("phase3", split_phase_dir / "phase3.txt", 2e-5, output_dir / "phase2" / "final"),
    ]
    for phase_name, image_list, lr, lora_path in phases:
        phase_dir = output_dir / phase_name
        cmd = [
            sys.executable,
            "kmc_lora/scripts/train_lora.py",
            "--base-model",
            str(model_cfg["base_model"]),
            "--image-list",
            str(image_list),
            "--output-dir",
            str(phase_dir),
            "--instance-prompt",
            str(prompts_cfg["instance_prompt"]),
            "--resolution",
            str(model_cfg["resolution"]),
            "--train-batch-size",
            str(model_cfg["train_batch_size"]),
            "--gradient-accumulation-steps",
            str(model_cfg["gradient_accumulation_steps"]),
            "--max-train-steps",
            str(step_plan[phase_name]),
            "--save-steps",
            str(max(0, min(int(model_cfg["save_steps"]), int(step_plan[phase_name])))),
            "--lr",
            str(lr),
            "--lora-rank",
            str(rank),
            "--lora-alpha",
            str(rank),
            "--seed",
            str(seed),
        ]
        validation_prompts = prompts_cfg.get("validation_prompts") or []
        if validation_prompts:
            cmd += ["--validation-prompts", *validation_prompts]
        if lora_path is not None:
            cmd += ["--lora-path", str(lora_path)]
        rc = run(cmd, log_file=phase_dir / "train.log", dry_run=dry_run)
        if rc != 0:
            return rc
    return 0


def final_adapter_dir(output_dir, spec_info):
    if spec_info["strategy"] == "kmc":
        return output_dir / "phase3" / "final"
    return output_dir / "final"


def generate_and_eval(cfg, spec_info, output_dir, experiment_name, seed, num_per_prompt, max_real, dry_run=False):
    prompts = cfg["prompts"]["validation_prompts"]
    gen_dir = output_dir / "generated"
    final_dir = final_adapter_dir(output_dir, spec_info)
    gen_cmd = [
        sys.executable,
        "kmc_lora/scripts/generate_samples.py",
        "--base-model",
        str(cfg["model"]["base_model"]),
        "--lora-path",
        str(final_dir),
        "--prompts",
        *prompts,
        "--out-dir",
        str(gen_dir),
        "--num-per-prompt",
        str(num_per_prompt),
        "--seed",
        str(seed),
    ]
    rc = run(gen_cmd, log_file=output_dir / "generate.log", dry_run=dry_run)
    if rc != 0:
        return rc

    real_list = get_eval_real_list(cfg, spec_info)
    eval_csv = output_dir / "eval.csv"
    eval_cmd = [
        sys.executable,
        "kmc_lora/scripts/evaluate_fid.py",
        "--real-list",
        str(real_list),
        "--gen-dir",
        str(gen_dir),
        "--prompts",
        *prompts,
        "--max-real",
        str(max_real),
        "--max-gen",
        str(num_per_prompt * len(prompts)),
        "--out-csv",
        str(eval_csv),
        "--experiment-name",
        experiment_name,
    ]
    return run(eval_cmd, log_file=output_dir / "eval.log", dry_run=dry_run)


def get_eval_real_list(cfg, spec_info):
    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    if spec_info["strategy"] in {"random", "kmc"}:
        return artifacts_dir / "splits" / f"{spec_info['split']}.txt"
    if spec_info["strategy"] == "anti":
        return artifacts_dir / "lists" / "AntiCurriculum.txt"
    if spec_info["strategy"] == "quality":
        return artifacts_dir / "lists" / "Quality-Top.txt"
    raise ValueError(f"unsupported strategy for eval list: {spec_info}")


def read_eval_metrics(output_dir):
    eval_csv = Path(output_dir) / "eval.csv"
    if eval_csv.exists():
        df = pd.read_csv(eval_csv)
        if not df.empty:
            row = df.iloc[-1]
            return {
                "fid": float(row["fid"]),
                "clip_score": float(row["clip_score"]) if str(row.get("clip_score", "")).strip() else None,
                "num_real": int(row.get("num_real", 0)) if "num_real" in row else None,
                "num_gen": int(row.get("num_gen", 0)) if "num_gen" in row else None,
            }

    # Fallback for interrupted CSV append cases: evaluate_fid.py always writes
    # generated/eval_metrics.json before trying to append eval.csv.
    eval_json = Path(output_dir) / "generated" / "eval_metrics.json"
    if eval_json.exists():
        try:
            payload = json.loads(eval_json.read_text(encoding="utf-8"))
            clip_raw = payload.get("clip_score", "")
            clip_score = float(clip_raw) if str(clip_raw).strip() else None
            return {
                "fid": float(payload["fid"]),
                "clip_score": clip_score,
                "num_real": int(payload.get("num_real", 0)) if payload.get("num_real", None) is not None else None,
                "num_gen": int(payload.get("num_gen", 0)) if payload.get("num_gen", None) is not None else None,
            }
        except Exception:
            return None

    return None


def append_run_log(log_csv, row):
    log_exists = log_csv.exists()
    with open(log_csv, "a" if log_exists else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "suite",
                "strategy",
                "split",
                "seed",
                "rank",
                "total_steps",
                "status",
                "elapsed_sec",
                "output_dir",
                "fid",
                "clip_score",
            ],
        )
        if not log_exists:
            writer.writeheader()
        writer.writerow(row)


def maybe_prepare_output_dir(output_dir, resume):
    if output_dir.exists() and not resume:
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)


def execute_one_run(cfg, dataset_name, suite_name, spec_info, seed, rank, total_steps, output_dir, args):
    metrics = read_eval_metrics(output_dir) if args.resume else None
    # When resuming, an existing eval.csv should be treated as a completed run
    # even if the current invocation skips train/generate/eval stages.
    if metrics and args.resume:
        return {"status": "cached", "metrics": metrics, "elapsed_sec": 0.0}

    if not args.dry_run:
        maybe_prepare_output_dir(output_dir, args.resume)
    manifest = {
        "dataset": dataset_name,
        "suite": suite_name,
        "strategy_spec": spec_info["raw"],
        "strategy_tag": strategy_tag(spec_info),
        "split": spec_info["split"],
        "seed": seed,
        "rank": rank,
        "total_steps": total_steps,
        "created_at_epoch": time.time(),
    }
    if not args.dry_run:
        (output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8"
        )

    t0 = time.time()
    rc = 0
    if not args.skip_train:
        if spec_info["strategy"] == "kmc":
            rc = train_kmc(cfg, spec_info["split"], output_dir, seed, total_steps, rank, dry_run=args.dry_run)
        else:
            image_list = get_single_list_path(cfg, spec_info)
            rc = train_single_phase(cfg, image_list, output_dir, seed, total_steps, rank, dry_run=args.dry_run)
    if rc == 0 and not args.skip_generate and not args.skip_eval:
        rc = generate_and_eval(
            cfg=cfg,
            spec_info=spec_info,
            output_dir=output_dir,
            experiment_name=output_dir.name,
            seed=seed,
            num_per_prompt=args.num_per_prompt,
            max_real=args.max_real,
            dry_run=args.dry_run,
        )
    elapsed = round(time.time() - t0, 1)
    metrics = read_eval_metrics(output_dir)
    return {
        "status": "ok" if (rc == 0 or metrics) else "fail",
        "metrics": metrics,
        "elapsed_sec": elapsed,
    }


def collect_records(out_root):
    records = []
    run_log = Path(out_root) / "run_log.csv"
    if not run_log.exists():
        return pd.DataFrame()
    return pd.read_csv(run_log)


def analyze_results(out_root):
    out_root = Path(out_root)
    analysis_dir = out_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    df = collect_records(out_root)
    if df.empty:
        print(f"[WARN] no run log found under {out_root}")
        return

    ok_df = df[df["status"].isin(["ok", "cached"])].copy()
    ok_df = ok_df[ok_df["fid"].notna()].copy()
    ok_df.to_csv(analysis_dir / "robustness_raw_metrics.csv", index=False)
    if ok_df.empty:
        print(f"[WARN] no completed metrics to analyze under {out_root}")
        return

    multi_df = ok_df[ok_df["suite"] == "multi_seed"].copy()
    if not multi_df.empty:
        multi_summary = (
            multi_df.groupby(["dataset", "strategy", "split"], dropna=False)
            .agg(
                n=("fid", "count"),
                fid_mean=("fid", "mean"),
                fid_std=("fid", "std"),
                fid_min=("fid", "min"),
                fid_max=("fid", "max"),
                clip_mean=("clip_score", "mean"),
                clip_std=("clip_score", "std"),
            )
            .reset_index()
        )
        multi_summary["fid_cv_pct"] = (multi_summary["fid_std"] / multi_summary["fid_mean"]) * 100.0
        multi_summary.to_csv(analysis_dir / "multi_seed_summary.csv", index=False)

        for dataset_name, group in multi_df.groupby("dataset"):
            fid_values = group["fid"].to_numpy()
            grand_mean = fid_values.mean()
            grouped = [sub["fid"].to_numpy() for _, sub in group.groupby("strategy")]
            ss_between = sum(len(vals) * (vals.mean() - grand_mean) ** 2 for vals in grouped)
            ss_within = sum(((vals - vals.mean()) ** 2).sum() for vals in grouped)
            ss_total = ss_between + ss_within
            variance_row = pd.DataFrame(
                [
                    {
                        "dataset": dataset_name,
                        "num_runs": int(len(group)),
                        "num_strategies": int(group["strategy"].nunique()),
                        "strategy_variance_ratio": ss_between / ss_total if ss_total else 0.0,
                        "randomness_variance_ratio": ss_within / ss_total if ss_total else 0.0,
                    }
                ]
            )
            variance_path = analysis_dir / f"multi_seed_variance_decomposition_{dataset_name}.csv"
            variance_row.to_csv(variance_path, index=False)

    hyper_df = ok_df[ok_df["suite"] == "hyperparam"].copy()
    if not hyper_df.empty:
        hyper_summary = (
            hyper_df.groupby(["dataset", "strategy", "split", "rank", "total_steps"], dropna=False)
            .agg(
                n=("fid", "count"),
                fid_mean=("fid", "mean"),
                fid_std=("fid", "std"),
                clip_mean=("clip_score", "mean"),
                clip_std=("clip_score", "std"),
            )
            .reset_index()
            .sort_values(["dataset", "strategy", "rank", "total_steps"])
        )
        hyper_summary["fid_cv_pct"] = (hyper_summary["fid_std"] / hyper_summary["fid_mean"]) * 100.0
        hyper_summary.to_csv(analysis_dir / "hyperparam_sensitivity_summary.csv", index=False)

        baseline_rows = hyper_summary.copy()
        baseline_rows["delta_from_best"] = baseline_rows.groupby("dataset")["fid_mean"].transform(
            lambda col: col - col.min()
        )
        baseline_rows.to_csv(analysis_dir / "hyperparam_sensitivity_delta.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description="Run multi-seed and hyperparameter robustness experiments")
    ap.add_argument("--dataset", required=True, choices=sorted(CONFIG_PATHS))
    ap.add_argument("--config-path", default=None,
                    help="Optional config override, useful for SDXL or custom experiment branches")
    ap.add_argument("--out-root", default="kmc_lora/results/robustness")
    ap.add_argument("--mode", choices=["all", "multi-seed", "hyperparam", "analyze-only"], default="all")
    ap.add_argument("--multi-seed-strategies", nargs="+", default=DEFAULT_MULTI_SEED_STRATEGIES)
    ap.add_argument("--multi-seeds", nargs="+", type=int, default=DEFAULT_MULTI_SEEDS)
    ap.add_argument("--hyperparam-strategy", default=None)
    ap.add_argument("--hyperparam-factor", choices=["rank", "steps", "both"], default="both")
    ap.add_argument("--hyperparam-seeds", nargs="+", type=int, default=DEFAULT_HPARAM_SEEDS)
    ap.add_argument("--rank-values", nargs="+", type=int, default=DEFAULT_RANKS)
    ap.add_argument("--step-values", nargs="+", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--num-per-prompt", type=int, default=50)
    ap.add_argument("--max-real", type=int, default=500)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--skip-analyze", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.config_path:
        config_path = args.config_path
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        config_path, cfg = load_cfg(args.dataset)
    out_root = Path(args.out_root) / args.dataset
    out_root.mkdir(parents=True, exist_ok=True)
    run_log = out_root / "run_log.csv"
    (out_root / "config_used.yaml").write_text(Path(config_path).read_text(encoding="utf-8"), encoding="utf-8")

    if args.mode == "analyze-only":
        analyze_results(out_root)
        return

    if args.mode in {"all", "multi-seed"}:
        for spec in args.multi_seed_strategies:
            spec_info = parse_strategy_spec(spec)
            tag = strategy_tag(spec_info)
            for seed in args.multi_seeds:
                output_dir = out_root / "multi_seed" / f"{tag}_seed{seed}"
                result = execute_one_run(
                    cfg=cfg,
                    dataset_name=args.dataset,
                    suite_name="multi_seed",
                    spec_info=spec_info,
                    seed=seed,
                    rank=int(cfg["model"]["lora_rank"]),
                    total_steps=int(cfg["model"]["max_train_steps"]),
                    output_dir=output_dir,
                    args=args,
                )
                if args.dry_run:
                    continue
                append_run_log(
                    run_log,
                    {
                        "dataset": args.dataset,
                        "suite": "multi_seed",
                        "strategy": tag,
                        "split": spec_info["split"] or "",
                        "seed": seed,
                        "rank": int(cfg["model"]["lora_rank"]),
                        "total_steps": int(cfg["model"]["max_train_steps"]),
                        "status": result["status"],
                        "elapsed_sec": result["elapsed_sec"],
                        "output_dir": str(output_dir),
                        "fid": result["metrics"]["fid"] if result["metrics"] else "",
                        "clip_score": result["metrics"]["clip_score"] if result["metrics"] else "",
                    },
                )
                if result["status"] == "fail":
                    raise SystemExit(1)

    if args.mode in {"all", "hyperparam"}:
        hyperparam_spec = parse_strategy_spec(args.hyperparam_strategy or DEFAULT_PARETO_STRATEGY[args.dataset])
        hyper_tag = strategy_tag(hyperparam_spec)
        default_rank = int(cfg["model"]["lora_rank"])
        default_steps = int(cfg["model"]["max_train_steps"])
        sweep_points = []
        if args.hyperparam_factor in {"rank", "both"}:
            for rank in args.rank_values:
                sweep_points.append(("rank", rank, default_steps))
        if args.hyperparam_factor in {"steps", "both"}:
            for total_steps in args.step_values:
                sweep_points.append(("steps", default_rank, total_steps))

        for factor_name, rank, total_steps in sweep_points:
            for seed in args.hyperparam_seeds:
                output_dir = out_root / "hyperparam" / factor_name / f"{hyper_tag}_rank{rank}_steps{total_steps}_seed{seed}"
                result = execute_one_run(
                    cfg=cfg,
                    dataset_name=args.dataset,
                    suite_name="hyperparam",
                    spec_info=hyperparam_spec,
                    seed=seed,
                    rank=rank,
                    total_steps=total_steps,
                    output_dir=output_dir,
                    args=args,
                )
                if args.dry_run:
                    continue
                append_run_log(
                    run_log,
                    {
                        "dataset": args.dataset,
                        "suite": "hyperparam",
                        "strategy": hyper_tag,
                        "split": hyperparam_spec["split"] or "",
                        "seed": seed,
                        "rank": rank,
                        "total_steps": total_steps,
                        "status": result["status"],
                        "elapsed_sec": result["elapsed_sec"],
                        "output_dir": str(output_dir),
                        "fid": result["metrics"]["fid"] if result["metrics"] else "",
                        "clip_score": result["metrics"]["clip_score"] if result["metrics"] else "",
                    },
                )
                if result["status"] == "fail":
                    raise SystemExit(1)

    if not args.skip_analyze and not args.dry_run:
        analyze_results(out_root)


if __name__ == "__main__":
    main()
