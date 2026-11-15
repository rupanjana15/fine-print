"""Fine Print: precompute the demo examples with the shipped 4-bit GGUF, on Kaggle CPU (D42).

One kernel per shard (SHARD is filled in by the copy in dist/kaggle/kernel-examples-<i>/), so three run side by
side on free CPU sessions. Same code path as the Space: `inference/pipeline.analyse_document` + `GGUFModel`.
Output: `fine-print-examples.zip` with data/examples/out/<slug>.json and the log.
"""

import glob
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

SHARD = "0/3"
W = "/kaggle/working"
REPO = f"{W}/fine-print"
src = next(os.path.dirname(p) for p in glob.glob("/kaggle/input/**/fine-print/configs", recursive=True))
shutil.copytree(src, REPO, dirs_exist_ok=True)
assert Path(f"{REPO}/data/examples/inputs.jsonl").exists(), "stale dataset version"
model = glob.glob("/kaggle/input/**/fine-print-v2-q4_k_m.gguf", recursive=True)[0]


def sh(cmd: str, log: str) -> None:
    with open(f"{W}/{log}", "a") as f:
        r = subprocess.run(cmd, shell=True, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, check=False,
                           env={**os.environ, "PYTHONPATH": f"{REPO}/src"})
    print(log, "exit", r.returncode, Path(f"{W}/{log}").read_text()[-1500:], flush=True)
    if r.returncode:
        raise RuntimeError(cmd)


sh("pip install -q langdetect llama-cpp-python==0.3.36 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu",
   "install.log")
sh(f"python -m fineprint.inference.examples run --model {model} --shard {SHARD} --threads {os.cpu_count() or 4}",
   "examples.log")

with zipfile.ZipFile(f"{W}/fine-print-examples.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob(f"{REPO}/data/examples/out/*.json"):
        z.write(f, os.path.relpath(f, REPO))
    for f in glob.glob(f"{W}/*.log"):
        z.write(f, "logs/kaggle_" + os.path.basename(f))
shutil.rmtree(REPO, ignore_errors=True)
