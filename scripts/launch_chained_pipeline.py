import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(r"e:\1 zhclian\AIGC 02")
PYTHON = Path(r"C:\Users\Administrator\AppData\Local\Programs\Python\Python311\python.exe")
LOGDIR = ROOT / "kmc_lora" / "results_structured" / "robustness" / "launcher_logs"
LOGDIR.mkdir(parents=True, exist_ok=True)
MASTER_LOG = LOGDIR / "current_robustness_master.log"
FAIL_LOG = LOGDIR / "current_robustness_failures.log"


def log(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with open(MASTER_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def log_failure(name: str, rc: int, attempt: int, args: list[str]):
    line = (
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
        f"FAIL {name} attempt={attempt} rc={rc} cmd={' '.join(args)}"
    )
    with open(FAIL_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_step(name: str, args: list[str], retries: int = 2, retry_wait_sec: int = 20):
    step_log = LOGDIR / f"{name}.log"
    for attempt in range(1, retries + 2):
        log(f"START {name} attempt={attempt}")
        log("CMD " + " ".join(args))
        with open(step_log, "a", encoding="utf-8") as f:
            f.write(f"\n===== ATTEMPT {attempt} @ {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            rc = subprocess.run(args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, text=True).returncode
        if rc == 0:
            log(f"END {name} attempt={attempt}")
            return
        log_failure(name, rc, attempt, args)
        if attempt <= retries:
            log(f"RETRY {name} after rc={rc}, waiting {retry_wait_sec}s")
            time.sleep(retry_wait_sec)
        else:
            log(f"FAIL {name} rc={rc}")
            raise SystemExit(rc)


def main():
    run_step(
        "salvage_anime_student_multiseed",
        [
            str(PYTHON),
            str(ROOT / "kmc_lora" / "scripts" / "run_robustness_suite.py"),
            "--dataset", "anime_student",
            "--mode", "multi-seed",
            "--out-root", str(ROOT / "kmc_lora" / "results_structured" / "robustness"),
            "--multi-seed-strategies", "random:D-High",
            "--multi-seeds", "42",
            "--resume",
            "--skip-train",
        ],
    )
    run_step(
        "salvage_wikiart_mixed_multiseed",
        [
            str(PYTHON),
            str(ROOT / "kmc_lora" / "scripts" / "run_robustness_suite.py"),
            "--dataset", "wikiart_mixed",
            "--mode", "multi-seed",
            "--out-root", str(ROOT / "kmc_lora" / "results_structured" / "robustness"),
            "--multi-seed-strategies", "random:D-High",
            "--multi-seeds", "42",
            "--resume",
            "--skip-train",
        ],
    )
    run_step(
        "continue_all_datasets_multiseed",
        [
            str(PYTHON),
            str(ROOT / "kmc_lora" / "scripts" / "run_robustness_all_datasets.py"),
            "--datasets", "anime_student", "wikiart_mixed", "dreambooth_mixed",
            "--mode", "multi-seed",
            "--out-root", str(ROOT / "kmc_lora" / "results_structured" / "robustness"),
            "--resume",
        ],
    )
    run_step(
        "experiment_plan_stage0",
        [
            str(PYTHON),
            str(ROOT / "kmc_lora" / "scripts" / "run_experiment_plan_stage0.py"),
        ],
    )
    run_step(
        "experiment_plan_stage1",
        [
            str(PYTHON),
            str(ROOT / "kmc_lora" / "scripts" / "run_experiment_plan_stage1.py"),
        ],
    )
    run_step(
        "experiment_plan_stage2",
        [
            str(PYTHON),
            str(ROOT / "kmc_lora" / "scripts" / "run_experiment_plan_stage2.py"),
        ],
    )
    log("COMPLETE")


if __name__ == "__main__":
    main()
