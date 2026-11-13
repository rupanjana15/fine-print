"""Fine Print: merge a LoRA adapter into the base model and convert it to GGUF Q4_K_M (decisions D18, D40).

Runs as a CPU-only Kaggle script (no GPU quota used): merging and quantising need RAM, not a GPU.
Inputs (attached as kernel sources):
- v1 adapter: output of the `fine-print-train` notebook (`ft/adapter` or `fine-print-adapter.zip`)
- v2 adapter: output of the `fine-print-eval` notebook, epoch-2 mode (`fine-print-adapter-v2.zip`)
Output: `fine-print-v1-q4_k_m.gguf`, `fine-print-v2-q4_k_m.gguf` (~1.9 GB each) and `gguf_info.json`.

Steps per adapter: base model in fp32 on CPU → merge LoRA (fp32 maths, then fp16 weights, like the HF eval)
→ llama.cpp `convert_hf_to_gguf.py` (f16 GGUF) → `llama-quantize` Q4_K_M. Intermediates are deleted as we go
(Kaggle's /kaggle/working is limited to ~20 GB).
"""

import glob
import json
import os
import shutil
import subprocess
import time

BASE = "Qwen/Qwen2.5-3B-Instruct"
W, TMP = "/kaggle/working", "/kaggle/tmp"
LLAMA = f"{TMP}/llama.cpp"
os.makedirs(TMP, exist_ok=True)


def sh(cmd: str) -> str:
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, check=False, shell=True, capture_output=True, text=True)
    print(r.stdout[-2000:], r.stderr[-2000:], flush=True)
    if r.returncode:
        raise RuntimeError(f"failed: {cmd}")
    return r.stdout


def find_adapter(name: str) -> str:
    """Unpack/locate an adapter folder (one that has adapter_config.json)."""
    out = f"{TMP}/adapter-{name}"
    if name == "v2":
        zips = glob.glob("/kaggle/input/**/fine-print-adapter-v2.zip", recursive=True)
        shutil.unpack_archive(zips[0], out)
        return out
    cfgs = [p for p in glob.glob("/kaggle/input/**/adapter_config.json", recursive=True) if "fine-print-train" in p]
    if cfgs:
        return os.path.dirname(cfgs[0])
    zips = [p for p in glob.glob("/kaggle/input/**/fine-print-adapter.zip", recursive=True)]
    shutil.unpack_archive(zips[0], out)
    return out


def merge(adapter: str, out: str) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.to(torch.float16).save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(BASE).save_pretrained(out)
    del model


def main() -> None:
    sh("ls -R /kaggle/input | head -60")
    sh("pip uninstall -y -q torchao; pip install -q 'peft>=0.17' gguf sentencepiece")
    sh(f"git clone --depth 1 https://github.com/ggml-org/llama.cpp {LLAMA}")
    commit = sh(f"git -C {LLAMA} rev-parse HEAD").strip()
    sh(f"cmake -S {LLAMA} -B {LLAMA}/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release "
       f"&& cmake --build {LLAMA}/build --target llama-quantize -j 4")
    quantize = glob.glob(f"{LLAMA}/build/**/llama-quantize", recursive=True)[0]

    info = {"llama_cpp_commit": commit, "base": BASE, "models": {}}
    for name in ("v2", "v1"):
        t = time.time()
        merged, f16 = f"{TMP}/merged-{name}", f"{TMP}/{name}-f16.gguf"
        q4 = f"{W}/fine-print-{name}-q4_k_m.gguf"
        merge(find_adapter(name), merged)
        sh(f"python {LLAMA}/convert_hf_to_gguf.py {merged} --outfile {f16} --outtype f16")
        shutil.rmtree(merged)
        sh(f"{quantize} {f16} {q4} Q4_K_M")
        os.remove(f16)
        info["models"][name] = {"file": os.path.basename(q4), "bytes": os.path.getsize(q4),
                                "minutes": round((time.time() - t) / 60, 1)}
        print(info, flush=True)
    with open(f"{W}/gguf_info.json", "w") as f:
        json.dump(info, f, indent=2)


main()
