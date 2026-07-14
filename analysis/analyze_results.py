#!/usr/bin/env python3
"""
Summarize RING / DP-FL experiment outputs.

This script understands both the new structured CSV logs produced by the updated
BD_Attack_Collusion.py and the older per-metric TXT files. Legacy TXT runs do
not contain latency, so latency summaries are only populated for runs generated
with the updated logger.
"""
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This analysis script requires pandas. Install it with `pip install pandas` "
        "or run it inside your existing scientific Python/Conda environment."
    ) from exc

ROUND_CSV_GLOB = "round_metrics_*.csv"
CLIENT_CSV_GLOB = "client_latency_*.csv"
LEGACY_PATTERNS = {
    "main_loss": re.compile(r"^main_loss_(\d+)\.txt$"),
    "main_acc_test": re.compile(r"^Main_Acc_test_(\d+)\.txt$"),
    "backdoor_acc_test": re.compile(r"^BD_Acc_test_(\d+)\.txt$"),
    "backdoor_loss": re.compile(r"^bd_loss_(\d+)\.txt$"),
}
EXPERIMENT_RE = re.compile(
    r"DP-SGD_Attack_(?P<attack_type>.*?)_Defense_(?P<defense>.*?)_frac=(?P<frac>.*?)_"
    r"nattacker=(?P<num_attacker>.*?)_iid_(?P<iid>.*?)_epsilon_(?P<dp_epsilon>.*?)_"
    r"clip_(?P<dp_clip>.*?)_lr_(?P<lr>.*?)_PDR_(?P<PDR>.*?)_local_ep_(?P<local_ep>.*?)_"
    r"CDP_(?P<central_noise>.*?)_random_drop_(?P<random_drop>.*)$"
)

GROUP_COLS = [
    "results_root", "dataset", "iid", "attack_type", "defense", "frac",
    "num_attacker", "dp_epsilon", "dp_clip", "lr", "PDR", "local_ep",
]
NUMERIC_COLS = [
    "round", "main_loss", "main_acc_test", "train_acc", "backdoor_loss",
    "backdoor_acc_test", "round_time_s", "benign_latency_mean_s",
    "malicious_latency_mean_s", "benign_latency_total_s",
    "malicious_latency_total_s", "defense_malicious_retention_rate",
    "defense_benign_retention_rate", "defense_selected_count",
    "collusion_adjustment_latency_s",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize experiment metrics across trials.")
    parser.add_argument(
        "--roots",
        nargs="+",
        default=["Results", "Results_DBA", "Results_Neuro"],
        help="Result roots to scan. Missing roots are skipped.",
    )
    parser.add_argument(
        "--out",
        default="analysis_summary",
        help="Directory where summary CSV files will be written.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Also write simple PNG plots for final accuracy and latency if matplotlib is installed.",
    )
    return parser.parse_args()


def read_csvs(paths: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            print(f"Skipping unreadable CSV {path}: {exc}")
            continue
        frame["source_file"] = str(path)
        frame["experiment_dir"] = str(path.parent)
        frame["results_root"] = path.parts[0] if not path.is_absolute() else _root_name(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _root_name(path: Path) -> str:
    for part in path.parts:
        if part in {"Results", "Results_DBA", "Results_Neuro"}:
            return part
    return path.anchor or "."


def read_number_lines(path: Path) -> List[Optional[float]]:
    values: List[Optional[float]] = []
    with path.open() as handle:
        for line in handle:
            text = line.strip()
            if not text:
                values.append(None)
                continue
            try:
                values.append(float(text))
            except ValueError:
                values.append(None)
    return values


def parse_experiment_metadata(exp_dir: Path, root: Path) -> Dict[str, object]:
    metadata: Dict[str, object] = {
        "results_root": root.name,
        "experiment_dir": str(exp_dir),
    }
    try:
        rel_parts = exp_dir.relative_to(root).parts
    except ValueError:
        rel_parts = exp_dir.parts
    if len(rel_parts) >= 1:
        metadata["dataset"] = rel_parts[0]
    if len(rel_parts) >= 2:
        metadata["iid"] = rel_parts[1]
    if rel_parts:
        match = EXPERIMENT_RE.match(rel_parts[-1])
        if match:
            metadata.update(match.groupdict())
    return metadata


def discover_legacy_rows(root: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for exp_dir in root.rglob("*"):
        if not exp_dir.is_dir():
            continue
        metric_files: Dict[int, Dict[str, Path]] = {}
        for txt in exp_dir.glob("*.txt"):
            for metric_name, pattern in LEGACY_PATTERNS.items():
                match = pattern.match(txt.name)
                if match:
                    run = int(match.group(1))
                    metric_files.setdefault(run, {})[metric_name] = txt
        if not metric_files:
            continue
        metadata = parse_experiment_metadata(exp_dir, root)
        for run, files in metric_files.items():
            series = {metric: read_number_lines(path) for metric, path in files.items()}
            rounds = max((len(values) for values in series.values()), default=0)
            for round_idx in range(rounds):
                row = dict(metadata)
                row.update({"run": run, "round": round_idx, "source_format": "legacy_txt"})
                for metric, values in series.items():
                    row[metric] = values[round_idx] if round_idx < len(values) else None
                rows.append(row)
    return pd.DataFrame(rows)


def normalize_rounds(rounds: pd.DataFrame) -> pd.DataFrame:
    if rounds.empty:
        return rounds
    rounds = rounds.copy()
    if "source_format" not in rounds.columns:
        rounds["source_format"] = "structured_csv"
    for col in GROUP_COLS:
        if col not in rounds.columns:
            rounds[col] = ""
    if "run" not in rounds.columns:
        rounds["run"] = rounds.get("source_file", "unknown")
    for col in NUMERIC_COLS:
        if col not in rounds.columns:
            rounds[col] = pd.NA
        rounds[col] = pd.to_numeric(rounds[col], errors="coerce")
    return rounds


def final_round_by_run(rounds: pd.DataFrame) -> pd.DataFrame:
    if rounds.empty:
        return rounds
    sort_cols = [c for c in GROUP_COLS + ["run", "round"] if c in rounds.columns]
    sorted_rounds = rounds.sort_values(sort_cols)
    idx_cols = GROUP_COLS + ["run"]
    idx = sorted_rounds.groupby(idx_cols, dropna=False)["round"].idxmax()
    return sorted_rounds.loc[idx].reset_index(drop=True)


def summarize(rounds: pd.DataFrame, finals: pd.DataFrame) -> pd.DataFrame:
    if finals.empty:
        return finals
    summary = finals.groupby(GROUP_COLS, dropna=False).agg(
        trials=("run", "nunique"),
        final_round_mean=("round", "mean"),
        final_main_acc_mean=("main_acc_test", "mean"),
        final_main_acc_std=("main_acc_test", "std"),
        final_backdoor_acc_mean=("backdoor_acc_test", "mean"),
        final_backdoor_acc_std=("backdoor_acc_test", "std"),
        final_main_loss_mean=("main_loss", "mean"),
        final_backdoor_loss_mean=("backdoor_loss", "mean"),
        final_benign_latency_mean_s=("benign_latency_mean_s", "mean"),
        final_malicious_latency_mean_s=("malicious_latency_mean_s", "mean"),
        final_benign_retention_mean=("defense_benign_retention_rate", "mean"),
        final_malicious_retention_mean=("defense_malicious_retention_rate", "mean"),
    ).reset_index()

    round_avgs = rounds.groupby(GROUP_COLS, dropna=False).agg(
        all_round_main_acc_mean=("main_acc_test", "mean"),
        all_round_backdoor_acc_mean=("backdoor_acc_test", "mean"),
        all_round_time_mean_s=("round_time_s", "mean"),
        all_round_benign_latency_mean_s=("benign_latency_mean_s", "mean"),
        all_round_malicious_latency_mean_s=("malicious_latency_mean_s", "mean"),
        all_round_benign_retention_mean=("defense_benign_retention_rate", "mean"),
        all_round_malicious_retention_mean=("defense_malicious_retention_rate", "mean"),
    ).reset_index()
    return summary.merge(round_avgs, on=GROUP_COLS, how="left")


def summarize_client_latency(client_csvs: List[Path]) -> pd.DataFrame:
    clients = read_csvs(client_csvs)
    if clients.empty:
        return clients
    for col in ["latency_s", "round", "run", "num_samples"]:
        if col in clients.columns:
            clients[col] = pd.to_numeric(clients[col], errors="coerce")
    group_cols = [c for c in ["results_root", "experiment_dir", "dataset", "iid", "attack_type", "defense", "role"] if c in clients.columns]
    return clients.groupby(group_cols, dropna=False).agg(
        observations=("latency_s", "count"),
        latency_mean_s=("latency_s", "mean"),
        latency_std_s=("latency_s", "std"),
        latency_median_s=("latency_s", "median"),
        latency_min_s=("latency_s", "min"),
        latency_max_s=("latency_s", "max"),
    ).reset_index()


def write_outputs(out_dir: Path, rounds: pd.DataFrame, finals: pd.DataFrame, setup_summary: pd.DataFrame,
                  latency_summary: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rounds.to_csv(out_dir / "round_metrics_all.csv", index=False)
    finals.to_csv(out_dir / "final_round_by_run.csv", index=False)
    setup_summary.to_csv(out_dir / "setup_summary.csv", index=False)
    if not latency_summary.empty:
        latency_summary.to_csv(out_dir / "client_latency_summary.csv", index=False)
    manifest = {
        "round_rows": int(len(rounds)),
        "final_run_rows": int(len(finals)),
        "setup_rows": int(len(setup_summary)),
        "latency_summary_rows": int(len(latency_summary)),
        "outputs": [
            "round_metrics_all.csv",
            "final_round_by_run.csv",
            "setup_summary.csv",
            "client_latency_summary.csv" if not latency_summary.empty else None,
        ],
    }
    with (out_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)


def write_plots(out_dir: Path, setup_summary: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plots.")
        return
    if setup_summary.empty:
        return
    plot_cols = [
        ("final_main_acc_mean", "final_main_accuracy_by_setup.png", "Final main accuracy"),
        ("final_backdoor_acc_mean", "final_backdoor_accuracy_by_setup.png", "Final backdoor accuracy"),
        ("final_benign_latency_mean_s", "final_benign_latency_by_setup.png", "Final benign latency (s)"),
        ("final_malicious_latency_mean_s", "final_malicious_latency_by_setup.png", "Final malicious latency (s)"),
    ]
    labels = setup_summary.apply(
        lambda r: f"{r.get('dataset','')}/{r.get('defense','')} eps={r.get('dp_epsilon','')} atk={r.get('num_attacker','')}",
        axis=1,
    )
    for col, filename, title in plot_cols:
        if col not in setup_summary.columns or setup_summary[col].dropna().empty:
            continue
        fig, ax = plt.subplots(figsize=(max(8, len(setup_summary) * 0.7), 5))
        ax.bar(range(len(setup_summary)), setup_summary[col])
        ax.set_title(title)
        ax.set_ylabel(col)
        ax.set_xticks(range(len(setup_summary)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=150)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    roots = [Path(root) for root in args.roots]
    existing_roots = [root for root in roots if root.exists()]
    if not existing_roots:
        raise SystemExit(f"None of the requested roots exist: {args.roots}")

    round_csvs: List[Path] = []
    client_csvs: List[Path] = []
    legacy_frames: List[pd.DataFrame] = []
    for root in existing_roots:
        round_csvs.extend(root.rglob(ROUND_CSV_GLOB))
        client_csvs.extend(root.rglob(CLIENT_CSV_GLOB))
        legacy_frames.append(discover_legacy_rows(root))

    structured = read_csvs(round_csvs)
    nonempty_legacy = [f for f in legacy_frames if not f.empty]
    legacy = pd.concat(nonempty_legacy, ignore_index=True) if nonempty_legacy else pd.DataFrame()
    rounds = normalize_rounds(pd.concat([structured, legacy], ignore_index=True, sort=False))
    if rounds.empty:
        raise SystemExit("No round_metrics CSV files or legacy metric TXT files were found.")

    finals = final_round_by_run(rounds)
    setup_summary = summarize(rounds, finals)
    latency = summarize_client_latency(client_csvs)
    out_dir = Path(args.out)
    write_outputs(out_dir, rounds, finals, setup_summary, latency)
    if args.plots:
        write_plots(out_dir, setup_summary)

    print(f"Wrote summaries to {out_dir}")
    print(f"Round rows: {len(rounds)} | final run rows: {len(finals)} | setups: {len(setup_summary)}")
    if legacy.shape[0] > 0:
        print("Note: legacy TXT rows do not include latency. Latency columns are populated only for new structured CSV runs.")


if __name__ == "__main__":
    main()
