from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "revision_20261006_r02"


def main():
    manifest = json.loads((RESULTS / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise RuntimeError("revision timing run is not complete")
    frame = pd.read_csv(RESULTS / "timing_summary.csv")
    setup, application = [], []
    for row in frame.itertuples():
        setup.append(
            f"{row.case} & {row.source_cells:,} & {row.target_cells:,} & "
            f"{1000 * row.gram_setup_median_ms:.2f} & {row.common_setup_median_ms:.2f} \\\\"
        )
        application.append(
            f"{row.case} & {row.rbf_prior_median_ms:.3f} & "
            f"{1000 * row.residual_median_ms:.2f} & {1000 * row.gram_solve_median_ms:.2f} & "
            f"{1000 * row.correction_update_median_ms:.2f} & "
            f"{row.projected_total_median_ms:.3f} & {row.common_apply_median_ms:.4f} \\\\"
        )
    for name, rows in (("runtime_setup_rows.tex", setup), ("runtime_application_rows.tex", application)):
        (ROOT / "tables" / name).write_text("\n".join(rows) + "\n\\bottomrule\n", encoding="ascii")
    print("Revision timing table rows regenerated from the completed run.")


if __name__ == "__main__":
    main()
