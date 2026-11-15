"""Fine Print: run the v2 model (+ threshold) over every chunk of every graded val/test service (decision D43).

Kaggle script, 2×T4: one shard per GPU, side by side. Same runner as the threshold study (`eval/run_knob.py`), with
`--threshold 0.9` so the expensive forced answer is only generated where it would be used.
Inputs: the `fine-print` dataset (code + data/examples/service_chunks.jsonl) and the v2 adapter from the
`fine-print-eval` notebook's output (`fine-print-adapter-v2.zip`).
Output: `fine-print-results.zip` with data/predictions/ft_qwen3b_v2_knob/services.shard{0,1}.jsonl and logs.
"""

import glob
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

W = "/kaggle/working"
REPO = f"{W}/fine-print"
src = next(os.path.dirname(p) for p in glob.glob("/kaggle/input/**/fine-print/configs", recursive=True))
shutil.copytree(src, REPO, dirs_exist_ok=True)
assert Path(f"{REPO}/data/examples/service_chunks.jsonl").exists(), "stale dataset version"
assert "--threshold" in Path(f"{REPO}/src/fineprint/eval/run_knob.py").read_text(), "stale dataset version"
ADAPTER = f"{W}/adapter-v2"
shutil.unpack_archive(glob.glob("/kaggle/input/**/fine-print-adapter-v2.zip", recursive=True)[0], ADAPTER)
ENV = {**os.environ, "PYTHONPATH": f"{REPO}/src", "TOKENIZERS_PARALLELISM": "false",
       "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

# torchao: Kaggle's old copy makes PEFT refuse to load an adapter on an fp16 model (D37).
r = subprocess.run('pip uninstall -y -q torchao; pip install -q "peft>=0.17"', shell=True, check=False)
print("install exit", r.returncode, flush=True)

cmd = (f"python -m fineprint.eval.run_knob --set services --system ft_qwen3b_v2_knob --adapter {ADAPTER} "
       "--threshold 0.9 --shard {i}/2")
jobs = {}
for i in (0, 1):
    log = open(f"{W}/services_gpu{i}.log", "w")  # noqa: SIM115 -- stays open while the job writes to it
    jobs[i] = subprocess.Popen(cmd.format(i=i), shell=True, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                               env={**ENV, "CUDA_VISIBLE_DEVICES": str(i)})
while any(p.poll() is None for p in jobs.values()):
    time.sleep(300)
    for i in jobs:
        lines = [x for x in Path(f"{W}/services_gpu{i}.log").read_text(errors="replace").splitlines() if x.strip()]
        print(f"gpu{i}:", lines[-1][:120] if lines else "(starting)", flush=True)
print({i: p.returncode for i, p in jobs.items()}, flush=True)

with zipfile.ZipFile(f"{W}/fine-print-results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob(f"{REPO}/data/predictions/*/services*.jsonl"):
        z.write(f, os.path.relpath(f, REPO))
    for f in glob.glob(f"{W}/*.log"):
        z.write(f, "logs/kaggle_" + os.path.basename(f))
shutil.rmtree(REPO, ignore_errors=True)
shutil.rmtree(ADAPTER, ignore_errors=True)
