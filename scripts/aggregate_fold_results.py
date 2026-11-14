from pathlib import Path
import argparse

import pandas as pd


def load_summary(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")
    return pd.read_csv(path)


def split_method_category(method: str) -> str:
    """'HAGI Regression Fold 0 - All' -> 'All'"""
    if " - " in method:
        return method.rsplit(" - ", 1)[-1]
    return method


def aggregate_rows(summaries):
    rows = []
    for fold, summary in summaries.items():
        for _, row in summary.iterrows():
            rows.append(
                {
                    "fold": fold,
                    "category": split_method_category(row["method"]),
                    "samples": row["samples"],
                    "mean_angular_error_deg": row["mean_angular_error_deg"],
                    "median_angular_error_deg": row["median_angular_error_deg"],
                    "std_angular_error_deg": row["std_angular_error_deg"],
                }
            )
    return pd.DataFrame(rows)


def summarize_across_folds(df: pd.DataFrame, model_label: str):
    grouped = df.groupby("category", as_index=False).agg(
        folds=("fold", "count"),
        mean_deg=("mean_angular_error_deg", "mean"),
        std_across_folds_deg=("mean_angular_error_deg", "std"),
        mean_median_deg=("median_angular_error_deg", "mean"),
        mean_std_deg=("std_angular_error_deg", "mean"),
    )
    grouped.insert(0, "model", model_label)
    return grouped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--baseline-prefix", default="hagi_regression_fold")
    parser.add_argument("--audio-prefix", default="hagi_audio_peak_fold")
    parser.add_argument("--audio-suffix", default="_uv", help="e.g. _uv for hagi_audio_peak_fold0_uv")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--out", default="results/five_fold_aggregate_summary.csv")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    baseline_summaries = {}
    audio_summaries = {}

    for fold in args.folds:
        baseline_path = results_root / f"{args.baseline_prefix}{fold}" / f"{args.baseline_prefix}{fold}_summary.csv"
        audio_name = f"{args.audio_prefix}{fold}{args.audio_suffix}"
        audio_path = results_root / audio_name / f"{audio_name}_summary.csv"
        baseline_summaries[fold] = load_summary(baseline_path)
        audio_summaries[fold] = load_summary(audio_path)

    baseline_agg = summarize_across_folds(aggregate_rows(baseline_summaries), "hagi_baseline")
    audio_agg = summarize_across_folds(aggregate_rows(audio_summaries), "hagi_audio_peak_uv")

    print("=== HAGI Baseline (mean +/- std across folds) ===")
    print(baseline_agg.to_string(index=False))
    print("\n=== HAGI + Peak AEM UV (mean +/- std across folds) ===")
    print(audio_agg.to_string(index=False))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    combined = pd.concat([baseline_agg, audio_agg], ignore_index=True)
    combined.to_csv(out_path, index=False)
    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
