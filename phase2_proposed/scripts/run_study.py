"""CLI: multi-dataset, multi-seed study. Runs the same experiments as
run_detection.py, phase1_baseline/scripts/run_remediation.py and
run_migration.py on several BEIR corpora and random seeds, so every headline
number can be reported as a mean and spread rather than one run.

Each (dataset, seed) gets its own config, derived from
phase2_proposed/configs/default.yaml with only the corpus, its size cap and
the seed changed, and writes three CSVs to
<out>/<dataset>/seed<seed>/: detection.csv, baseline.csv and phase2.csv.
A finished run is skipped, so the study can be stopped and resumed.
aggregate_study.py turns the CSVs into tables and figures.

The seed changes the corpus subsample (where one is taken), the Zipfian
query popularity, the contamination masks and the probe batch.

Usage:
    python phase2_proposed/scripts/run_study.py [--datasets scifact nfcorpus ...] [--seeds 42 43 44]
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _src in ("common/src", "phase1_baseline/src", "phase2_proposed/src"):
    sys.path.insert(0, str(_REPO_ROOT / _src))

import pandas as pd  # noqa: E402

from common.config import Config, load_config  # noqa: E402
from common.seed import set_seed  # noqa: E402

# Corpus size cap per dataset: large enough to keep every judged document
# (see common.data.subsample_corpus), padded with random distractors.
DATASETS = {
    "scifact": 5_000,  # 667 judged documents
    "nfcorpus": 5_000,  # whole corpus (3,633 documents, all judged)
    "fiqa": 20_000,  # 17,110 judged documents
    "quora": 25_000,  # 23,301 judged documents
}


def _load_script(relative_path: str):
    spec = importlib.util.spec_from_file_location(Path(relative_path).stem, _REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def study_config(base: Config, dataset: str, seed: int) -> Config:
    return dataclasses.replace(
        base, seed=seed, corpus=dataclasses.replace(base.corpus, name=dataset, max_docs=DATASETS[dataset])
    )


def run_one(cfg: Config, out_dir: Path, detection, remediation, migration) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = [
        ("detection.csv", lambda: pd.DataFrame(detection.run_config(cfg.corpus.name, cfg))),
        ("baseline.csv", lambda: remediation.run_remediation_sweep(cfg)),
        ("phase2.csv", lambda: migration.run_migration_sweep(cfg)),
    ]
    for filename, step in steps:
        path = out_dir / filename
        if path.exists():
            print(f"  {path} exists, skipping")
            continue
        t0 = time.time()
        set_seed(cfg.seed)
        df = step()
        df.insert(0, "seed", cfg.seed)
        df.insert(0, "dataset", cfg.corpus.name)
        df.to_csv(path, index=False)
        print(f"  saved {path} ({time.time() - t0:.0f}s)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="phase2_proposed/configs/default.yaml")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--out", default="phase2_proposed/results/study")
    args = parser.parse_args()

    detection = _load_script("phase2_proposed/scripts/run_detection.py")
    remediation = _load_script("phase1_baseline/scripts/run_remediation.py")
    migration = _load_script("phase2_proposed/scripts/run_migration.py")
    base = load_config(args.config)

    for dataset in args.datasets:
        for seed in args.seeds:
            print(f"=== {dataset} seed {seed} ===", flush=True)
            t0 = time.time()
            run_one(study_config(base, dataset, seed), Path(args.out) / dataset / f"seed{seed}",
                    detection, remediation, migration)
            print(f"=== {dataset} seed {seed} done in {time.time() - t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    main()
