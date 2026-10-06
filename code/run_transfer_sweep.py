from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete transfer neighborhood and metric study.")
    parser.add_argument("--surface-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "transfer_sweep")
    parser.add_argument("--parallel-runs", type=int, default=0)
    parser.add_argument("--total-workers", type=int, default=0)
    args = parser.parse_args()

    total_workers = args.total_workers or int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    studies = [(value, "capacity") for value in (1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64)]
    studies.extend([(24, "uniform"), (24, "surface_area")])
    parallel_runs = args.parallel_runs or min(len(studies), max(total_workers // 4, 1))
    workers_per_run = max(total_workers // parallel_runs, 1)
    args.output.mkdir(parents=True, exist_ok=True)

    records = []
    with ThreadPoolExecutor(max_workers=parallel_runs) as executor:
        futures = {
            executor.submit(
                _run_one,
                args.surface_root,
                args.output,
                neighbors,
                metric,
                workers_per_run,
            ): (neighbors, metric)
            for neighbors, metric in studies
        }
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps(record), flush=True)

    summary_rows = []
    for record in sorted(records, key=lambda item: (item["neighbors"], item["projection_metric"])):
        frame = pd.read_csv(Path(record["output"]) / "aggregate_metrics.csv")
        frame.insert(0, "projection_metric", record["projection_metric"])
        frame.insert(0, "neighbors", record["neighbors"])
        summary_rows.append(frame)
    pd.concat(summary_rows, ignore_index=True).to_csv(args.output / "sweep_aggregate_metrics.csv", index=False)
    (args.output / "sweep_manifest.json").write_text(
        json.dumps(
            {
                "study_count": len(studies),
                "parallel_runs": parallel_runs,
                "total_workers": total_workers,
                "workers_per_run": workers_per_run,
                "studies": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _run_one(
    surface_root: Path,
    output_root: Path,
    neighbors: int,
    metric: str,
    workers: int,
) -> dict:
    destination = output_root / f"k_{neighbors:02d}_{metric}"
    command = [
        sys.executable,
        str(ROOT / "code" / "analyze_transfer.py"),
        "--surface-root",
        str(surface_root),
        "--output",
        str(destination),
        "--neighbors",
        str(neighbors),
        "--workers",
        str(workers),
        "--projection-metric",
        metric,
        "--duration-s",
        "300",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return {
        "neighbors": neighbors,
        "projection_metric": metric,
        "workers": workers,
        "output": str(destination),
        "last_output_line": completed.stdout.strip().splitlines()[-1],
    }


if __name__ == "__main__":
    main()
