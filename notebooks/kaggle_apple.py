"""Fine Print: run the shipped 4-bit GGUF on Apple's iTunes terms (ToS;DR document 225), on Kaggle CPU.

A one-off check of the extension's result against ToS;DR's human points (results/apple_check.md), run here
because the Mac overheats on long documents. Same code path as the extension: `inference.pipeline` + `GGUFModel`.
Output: `fine-print-apple.zip` with apple_itunes_result.json and the log.
"""

import glob
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

W = "/kaggle/working"
REPO = f"{W}/fine-print"
src = next(os.path.dirname(p) for p in glob.glob("/kaggle/input/**/fine-print/configs", recursive=True))
shutil.copytree(src, REPO, dirs_exist_ok=True)
inp = Path(f"{REPO}/data/examples/apple_itunes.jsonl")
assert inp.exists(), "stale dataset version"
Path(f"{W}/apple_itunes.txt").write_text(json.loads(inp.read_text())["text"])
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
sh(f"python -m fineprint.inference.pipeline --model {model} --file {W}/apple_itunes.txt "
   f"--out {W}/apple_itunes_result.json", "apple.log")

with zipfile.ZipFile(f"{W}/fine-print-apple.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.write(f"{W}/apple_itunes_result.json", "apple_itunes_result.json")
    for f in glob.glob(f"{W}/*.log"):
        z.write(f, "logs/kaggle_" + os.path.basename(f))
shutil.rmtree(REPO, ignore_errors=True)
