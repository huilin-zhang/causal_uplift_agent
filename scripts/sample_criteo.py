"""Stream the 13.9M-row Criteo file in chunks and keep a small random sample.

    python -m scripts.sample_criteo --source path/to/criteo-uplift-v2.1.csv.gz --rate 0.005

The full file is ~3 GB and never enters the repository. The sample goes to
data/external/criteo/criteo_sample.csv.gz (git-ignored).
Code guide: section "Real datasets" (sec:real).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from app.data.real_datasets import data_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--rate", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    parts = []
    for chunk in pd.read_csv(args.source, chunksize=500_000):
        keep = rng.random(len(chunk)) < args.rate  # Bernoulli sample keeps arm shares unbiased
        parts.append(chunk[keep])
    sample = pd.concat(parts, ignore_index=True)
    out = data_dir() / "criteo"
    out.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out / "criteo_sample.csv.gz", index=False)
    print(f"kept {len(sample):,} rows -> {out / 'criteo_sample.csv.gz'}")


if __name__ == "__main__":
    main()
