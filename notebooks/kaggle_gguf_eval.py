"""Fine Print: score the shipped 4-bit GGUF (v2 + empty-answer threshold) on test-500, on Kaggle CPU (D40).

Runs the exact Mac code path (`eval/run_gguf.py --threshold`, llama-cpp-python) on a CPU-only Kaggle session,
so no GPU quota is used and the author's laptop stays cool. Same model file, same code; only the hardware
differs, which changes speed, not answers (checked against the chunks first scored on the Mac).
Inputs: the `fine-print` dataset (code + data) and the `fine-print-gguf` script output (the .gguf file).
Output: `fine-print-results.zip` with predictions, scores and the log.
"""

import glob
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

W = "/kaggle/working"
REPO = f"{W}/fine-print"
src = next(os.path.dirname(p) for p in glob.glob("/kaggle/input/**/fine-print/configs", recursive=True))
shutil.copytree(src, REPO, dirs_exist_ok=True)
assert "def analyse" in Path(f"{REPO}/src/fineprint/inference/gguf.py").read_text(), "stale dataset version"
model = glob.glob("/kaggle/input/**/fine-print-v2-q4_k_m.gguf", recursive=True)[0]


def sh(cmd: str, log: str) -> None:
    with open(f"{W}/{log}", "a") as f:
        r = subprocess.run(cmd, shell=True, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, check=False,
                           env={**os.environ, "PYTHONPATH": f"{REPO}/src"})
    print(log, "exit", r.returncode, Path(f"{W}/{log}").read_text()[-1500:], flush=True)
    if r.returncode:
        raise RuntimeError(cmd)


# Prebuilt CPU wheel if there is one for this Python, else pip builds llama.cpp from source (~10 min).
sh("pip install -q llama-cpp-python==0.3.36 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu"
   " && python -c 'import llama_cpp; print(llama_cpp.__version__)'", "install.log")
threads = os.cpu_count() or 4
sh(f"python -m fineprint.eval.run_gguf --system ft_qwen3b_v2_gguf --set test-500 --model {model} "
   f"--threshold 0.9 --threads {threads}", "gguf_eval.log")

with zipfile.ZipFile(f"{W}/fine-print-results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for pattern in ["results/eval/*.json", "data/predictions/*/*.jsonl"]:
        for f in glob.glob(f"{REPO}/{pattern}"):
            z.write(f, os.path.relpath(f, REPO))
    for f in glob.glob(f"{W}/*.log"):
        z.write(f, "logs/kaggle_" + os.path.basename(f))
shutil.rmtree(REPO, ignore_errors=True)
