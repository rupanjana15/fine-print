"""Import fine-print-results.zip from the Kaggle notebook and re-score it on the Mac (decision D35).

Re-scoring locally (instead of trusting the JSON made on Kaggle) checks that both sides used the
same eval sets: the set fingerprints must match.

Usage:  uv run python -m fineprint.kaggle_results [path/to/fine-print-results.zip]
"""

import json
import sys
import zipfile

from fineprint.config import ROOT, data_path
from fineprint.eval.score import build_table, score_run


def main() -> None:
    zip_path = sys.argv[1] if len(sys.argv) > 1 else ROOT / "fine-print-results.zip"
    runs = []
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.startswith("data/predictions/"):
                system, file = name.split("/")[2:4]
                target = data_path("predictions") / system / file
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(name))
                runs.append((system, file.removesuffix(".jsonl")))
            elif name.startswith("logs/"):
                target = data_path("predictions") / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(name))
        kaggle_results = {n: json.loads(z.read(n)) for n in z.namelist() if n.startswith("results/eval/")}

    for system, eval_set in sorted(runs):
        remote = kaggle_results.get(f"results/eval/{system}__{eval_set}.json", {})
        notes = remote.get("notes", "run on Kaggle")
        local = score_run(system, eval_set, notes=notes)
        if remote and remote["set_fingerprint"] != local["set_fingerprint"]:
            print(f"WARNING {system}/{eval_set}: Kaggle used a different eval set "
                  f"({remote['set_fingerprint']} vs {local['set_fingerprint']}); rebuild the bundle")
        if remote.get("seconds_per_example") and not local["seconds_per_example"]:
            local["seconds_per_example"] = remote["seconds_per_example"]
        print(f"{system:24} {eval_set:9} detect F1 {local['detection']['f1']:.3f} "
              f"macro-F1 {local['category_macro_f1']:.3f}")
    print(build_table())


if __name__ == "__main__":
    main()
