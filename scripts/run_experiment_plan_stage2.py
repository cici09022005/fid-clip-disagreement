import json
import subprocess
import time
from pathlib import Path


ROOT = Path(r"e:\1 zhclian\AIGC 02")
PYTHON = Path(r"C:\Users\Administrator\AppData\Local\Programs\Python\Python311\python.exe")
TRACKER = ROOT / "refine-logs" / "EXPERIMENT_TRACKER.md"
OUT_DIR = ROOT / "kmc_lora" / "figures" / "experiment_plan_m2"
SDXL_ROOT = ROOT / "models" / "sdxl_base_1.0"


def update_tracker(run_ids, status, note_suffix=""):
    if not TRACKER.exists():
        return
    text = TRACKER.read_text(encoding="utf-8")
    for run_id in run_ids:
        lines = []
        for line in text.splitlines():
            if line.startswith(f"| {run_id} |"):
                parts = line.split("|")
                if len(parts) >= 10:
                    parts[8] = f" {status} "
                    if note_suffix:
                        parts[9] = f" {note_suffix} "
                    line = "|".join(parts)
            lines.append(line)
        text = "\n".join(lines)
    TRACKER.write_text(text, encoding="utf-8")


def model_ready():
    required = [
        SDXL_ROOT / "model_index.json",
        SDXL_ROOT / "unet" / "config.json",
        SDXL_ROOT / "text_encoder" / "config.json",
        SDXL_ROOT / "text_encoder_2" / "config.json",
        SDXL_ROOT / "tokenizer" / "tokenizer_config.json",
        SDXL_ROOT / "tokenizer_2" / "tokenizer_config.json",
    ]
    if not all(p.exists() for p in required):
        return False
    unet_files = list((SDXL_ROOT / "unet").glob("*.safetensors")) + list((SDXL_ROOT / "unet").glob("*.bin"))
    vae_dir = SDXL_ROOT / "vae"
    vae_files = list(vae_dir.glob("*.safetensors")) + list(vae_dir.glob("*.bin")) if vae_dir.exists() else []
    return bool(unet_files) and bool(vae_files)


def run_step(name, cmd):
    log_path = OUT_DIR / f"{name}.log"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("CMD " + " ".join(cmd) + "\n")
        rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, text=True).returncode
    if rc != 0:
        raise SystemExit(rc)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    update_tracker(["R008", "R009", "R010", "R011"], "IN_PROGRESS", "auto-stage2 waiting for SDXL and launching M2")

    deadline = time.time() + 12 * 3600
    while time.time() < deadline:
        if model_ready():
            break
        time.sleep(300)

    ready = model_ready()
    (OUT_DIR / "sdxl_readiness.json").write_text(json.dumps({"ready": ready}, indent=2), encoding="utf-8")
    if not ready:
        update_tracker(["R008", "R009"], "TODO", "SDXL model not fully ready within wait window")
        print("SDXL model not ready within wait window.")
        return

    run_step(
        "sdxl_wikiart_anchor",
        [
            str(PYTHON),
            str(ROOT / "kmc_lora" / "scripts" / "run_robustness_suite.py"),
            "--dataset", "wikiart_mixed",
            "--config-path", str(ROOT / "kmc_lora" / "configs" / "wikiart_mixed_sdxl_local.yaml"),
            "--mode", "multi-seed",
            "--out-root", str(ROOT / "kmc_lora" / "results" / "robustness_sdxl_anchor"),
            "--multi-seed-strategies", "random:D-Low", "kmc:D-Low", "anti", "random:D-Sub-50", "kmc:D-Sub-50",
            "--multi-seeds", "42", "52", "62",
            "--num-per-prompt", "30",
            "--max-real", "300",
            "--resume",
        ],
    )
    run_step(
        "sdxl_dreambooth_mixed_anchor",
        [
            str(PYTHON),
            str(ROOT / "kmc_lora" / "scripts" / "run_robustness_suite.py"),
            "--dataset", "dreambooth_mixed",
            "--config-path", str(ROOT / "kmc_lora" / "configs" / "dreambooth_mixed_sdxl_local.yaml"),
            "--mode", "multi-seed",
            "--out-root", str(ROOT / "kmc_lora" / "results" / "robustness_sdxl_anchor"),
            "--multi-seed-strategies", "random:D-Low", "kmc:D-Low", "anti", "random:D-Sub-50", "kmc:D-Sub-50",
            "--multi-seeds", "42", "52", "62",
            "--num-per-prompt", "30",
            "--max-real", "300",
            "--resume",
        ],
    )
    run_step(
        "sd15_wikiart_hparam",
        [
            str(PYTHON),
            str(ROOT / "kmc_lora" / "scripts" / "run_robustness_suite.py"),
            "--dataset", "wikiart_mixed",
            "--mode", "hyperparam",
            "--out-root", str(ROOT / "kmc_lora" / "results" / "robustness_hparam"),
            "--hyperparam-strategy", "random:D-Low",
            "--hyperparam-factor", "both",
            "--rank-values", "4", "16", "64",
            "--step-values", "600", "1200", "2400",
            "--hyperparam-seeds", "42", "52", "62",
            "--num-per-prompt", "30",
            "--max-real", "300",
            "--resume",
        ],
    )
    run_step(
        "sd15_dreambooth_mixed_hparam",
        [
            str(PYTHON),
            str(ROOT / "kmc_lora" / "scripts" / "run_robustness_suite.py"),
            "--dataset", "dreambooth_mixed",
            "--mode", "hyperparam",
            "--out-root", str(ROOT / "kmc_lora" / "results" / "robustness_hparam"),
            "--hyperparam-strategy", "random:D-Low",
            "--hyperparam-factor", "both",
            "--rank-values", "4", "16", "64",
            "--step-values", "600", "1200", "2400",
            "--hyperparam-seeds", "42", "52", "62",
            "--num-per-prompt", "30",
            "--max-real", "300",
            "--resume",
        ],
    )

    update_tracker(["R008", "R009", "R010", "R011"], "DONE", "auto-stage2 launched SDXL anchor and SD1.5 hyperparam")
    print("Stage2 automation finished.")


if __name__ == "__main__":
    main()
