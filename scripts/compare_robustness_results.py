#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

VARIANTS = ("centroid_gated", "argmax_gated", "centroid_concat")


def load_fold_mean(results_root: Path, variant: str, fold: int = 0) -> float | None:
    name = f"hagi_audio_peak_uv_{variant}_fold{fold}"
    summary = results_root / name / f"{name}_summary.csv"
    if not summary.exists():
        return None
    with summary.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["method"].endswith(" - All"):
                return float(row["mean_angular_error_deg"])
    return None


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default=str(repo_root / "results"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--baseline-mean", type=float, default=10.356707)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    rows = []
    for variant in VARIANTS:
        mean_deg = load_fold_mean(results_root, variant, fold=args.fold)
        if mean_deg is None:
            rows.append({"variant": variant, "mean_deg_all": "", "status": "missing"})
        else:
            rows.append({"variant": variant, "mean_deg_all": f"{mean_deg:.6f}", "status": "ok"})

    print(f"=== robustness fold {args.fold} (all gaze) ===")
    for row in rows:
        if row["status"] == "ok":
            print(f"  {row['variant']:18s}  {float(row['mean_deg_all']):.3f} deg")
        else:
            print(f"  {row['variant']:18s}  missing")

    ok = {r["variant"]: float(r["mean_deg_all"]) for r in rows if r["status"] == "ok"}
    if len(ok) >= 2:
        if "centroid_gated" in ok and "argmax_gated" in ok:
            d = ok["centroid_gated"] - ok["argmax_gated"]
            print(f"centroid_gated vs argmax_gated: {d:+.3f} deg")
        if "centroid_gated" in ok and "centroid_concat" in ok:
            d = ok["centroid_gated"] - ok["centroid_concat"]
            print(f"centroid_gated vs centroid_concat: {d:+.3f} deg")

    if ok:
        winner = min(ok, key=ok.get)
        print(f"best variant: {winner} ({ok[winner]:.3f} deg)")
        print(f"primary peak UV: {args.baseline_mean:.3f} deg ({ok[winner] - args.baseline_mean:+.3f} deg)")

    out = results_root / "robustness_fold0_comparison.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "mean_deg_all", "status"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
