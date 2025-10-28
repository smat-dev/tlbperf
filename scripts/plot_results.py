#!/usr/bin/env python3
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
SUMMARY_PATH = DATA_DIR / "summary.json"


def format_working_set(bytes_val: int) -> str:
    gib = 1024**3
    mib = 1024**2
    kib = 1024
    if bytes_val >= gib:
        value = bytes_val // gib
        return f"{value} GiB"
    if bytes_val >= mib:
        value = bytes_val // mib
        return f"{value} MiB"
    value = bytes_val // kib
    return f"{value} KiB"


def parse_meta(line: str) -> dict[str, Any]:
    line = line.strip()
    if line.startswith("#"):
        line = line[1:].strip()
    meta: dict[str, Any] = {}
    for token in line.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if value.isdigit():
            meta[key] = int(value)
        else:
            try:
                meta[key] = float(value)
            except ValueError:
                if value.lower() in {"true", "false"}:
                    meta[key] = value.lower() == "true"
                else:
                    meta[key] = value
    return meta


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[pd.DataFrame] = []
    run_records: list[dict[str, Any]] = []
    for path in sorted(DATA_DIR.glob("*.csv")):
        with path.open() as fh:
            first_line = fh.readline()
        meta = parse_meta(first_line)
        df = pd.read_csv(path, comment="#")
        if df.empty:
            continue
        df["mode"] = meta.get("mode", "")
        df["vertices"] = int(meta.get("vertices", 0))
        df["steps"] = int(meta.get("steps", 0))
        df["block_vertices"] = int(meta.get("block_vertices", 0))
        df["block_steps"] = int(meta.get("block_steps", 0))
        df["prefetch_distance"] = int(meta.get("prefetch_distance", 0))
        df["advise_huge"] = bool(meta.get("advise_huge", False))
        df["advise_nohuge"] = bool(meta.get("advise_nohuge", False))
        df["ns_per_cycle"] = float(meta.get("ns_per_cycle", math.nan))
        df["filename"] = path.name
        hugetlb_mode = str(meta.get("hugetlb_mode", "none"))
        df["hugetlb_mode"] = hugetlb_mode
        if hugetlb_mode == "2m":
            variant = "hugetlb_2m"
        elif hugetlb_mode == "1g":
            variant = "hugetlb_1g"
        elif df["advise_huge"].iloc[0]:
            variant = "thp_hint"
        elif df["advise_nohuge"].iloc[0]:
            variant = "nohuge"
        else:
            variant = "standard"
        df["variant"] = variant
        df["working_set_bytes"] = df["vertices"] * 64
        records.append(df)
        run_records.append(
            {
                "filename": path.name,
                "mode": df["mode"].iloc[0],
                "variant": df["variant"].iloc[0],
                "vertices": int(df["vertices"].iloc[0]),
                "working_set_bytes": int(df["working_set_bytes"].iloc[0]),
                "steps": int(df["steps"].iloc[0]),
                "ns_per_cycle": float(df["ns_per_cycle"].iloc[0]),
            }
        )
    if not records:
        raise SystemExit("No CSV files found in data directory")
    return pd.concat(records, ignore_index=True), pd.DataFrame(run_records)


def compute_quantiles(df: pd.DataFrame, quantiles: list[float], labels: dict[float, str]) -> pd.DataFrame:
    q_df = (
        df.groupby(["mode", "variant", "vertices", "working_set_bytes"])["nanoseconds"]
        .quantile(quantiles)
        .reset_index()
    )
    q_df.rename(columns={"level_4": "quantile"}, inplace=True)
    q_df["quantile"] = q_df["quantile"].round(4)
    q_df["quantile_label"] = q_df["quantile"].map(labels)
    q_df = q_df.dropna(subset=["quantile_label"])
    q_df["working_set_mb"] = q_df["working_set_bytes"] / (1024**2)
    return q_df


def plot_baseline_quantiles(q_df: pd.DataFrame) -> None:
    baseline = q_df[q_df["mode"] == "baseline"].copy()
    baseline["variant_label"] = baseline["variant"].map(
        {
            "standard": "4 KiB pages",
            "hugetlb_2m": "2 MiB pages",
            "hugetlb_1g": "1 GiB pages",
        }
    )
    baseline = baseline.dropna(subset=["variant_label"])
    baseline["quantile_label"] = pd.Categorical(
        baseline["quantile_label"],
        categories=["P50", "P95"],
        ordered=True,
    )
    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=baseline,
        x="working_set_mb",
        y="nanoseconds",
        hue="quantile_label",
        style="variant_label",
        markers=True,
        dashes=False,
    )
    ax = plt.gca()
    ax.set_xscale("log", base=2)
    unique_bytes = sorted(baseline["working_set_bytes"].unique())
    xticks = [b / (1024**2) for b in unique_bytes]
    ax.set_xticks(xticks)
    ax.set_xticklabels([format_working_set(int(b)) for b in unique_bytes], rotation=30, ha="right")
    plt.xlabel("Working set size")
    plt.ylabel("Per-hop latency (ns)")
    plt.title("Pointer-chase latency vs. working set size")
    plt.grid(True, which="both", linestyle=":", linewidth=0.6)
    plt.tight_layout()
    out_path = FIG_DIR / "baseline_quantiles.png"
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_baseline_standard_quantiles(df: pd.DataFrame) -> None:
    baseline = df[(df["mode"] == "baseline") & (df["variant"] == "standard")]
    if baseline.empty:
        return

    labels = {0.5: "P50", 0.95: "P95"}
    q_df = compute_quantiles(baseline, list(labels.keys()), labels)
    q_df["quantile_label"] = pd.Categorical(
        q_df["quantile_label"],
        categories=["P50", "P95"],
        ordered=True,
    )
    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=q_df,
        x="working_set_mb",
        y="nanoseconds",
        hue="quantile_label",
        markers=True,
        dashes=False,
    )
    ax = plt.gca()
    ax.set_xscale("log", base=2)
    unique_bytes = sorted(q_df["working_set_bytes"].unique())
    xticks = [b / (1024**2) for b in unique_bytes]
    ax.set_xticks(xticks)
    ax.set_xticklabels([format_working_set(int(b)) for b in unique_bytes], rotation=30, ha="right")
    plt.xlabel("Working set size")
    plt.ylabel("Per-hop latency (ns)")
    plt.title("Pointer-chase latency (4 KiB pages)")
    plt.grid(True, which="both", linestyle=":", linewidth=0.6)
    plt.tight_layout()
    out_path = FIG_DIR / "baseline_standard_quantiles.png"
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_baseline_2m_quantiles(df: pd.DataFrame) -> None:
    baseline = df[(df["mode"] == "baseline") & (df["variant"] == "hugetlb_2m")]
    if baseline.empty:
        return

    labels = {0.5: "P50", 0.95: "P95"}
    q_df = compute_quantiles(baseline, list(labels.keys()), labels)
    q_df["quantile_label"] = pd.Categorical(
        q_df["quantile_label"],
        categories=["P50", "P95"],
        ordered=True,
    )
    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=q_df,
        x="working_set_mb",
        y="nanoseconds",
        hue="quantile_label",
        markers=True,
        dashes=False,
    )
    ax = plt.gca()
    ax.set_xscale("log", base=2)
    unique_bytes = sorted(q_df["working_set_bytes"].unique())
    xticks = [b / (1024**2) for b in unique_bytes]
    ax.set_xticks(xticks)
    ax.set_xticklabels([format_working_set(int(b)) for b in unique_bytes], rotation=30, ha="right")
    plt.xlabel("Working set size")
    plt.ylabel("Per-hop latency (ns)")
    plt.title("Pointer-chase latency (2 MiB pages)")
    plt.grid(True, which="both", linestyle=":", linewidth=0.6)
    plt.tight_layout()
    out_path = FIG_DIR / "baseline_2m_quantiles.png"
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_mitigation_bars(df: pd.DataFrame, quantiles: list[float]) -> dict[str, Any]:
    baseline_vertices = sorted(df[df["mode"] == "baseline"]["vertices"].unique())
    if not baseline_vertices:
        return {}
    required_strategies = [
        ("baseline", "standard"),
        ("baseline", "hugetlb_2m"),
        ("baseline", "hugetlb_1g"),
    ]

    has_blocked = not df[df["mode"] == "blocked"].empty
    if has_blocked:
        required_strategies.append(("blocked", "standard"))

    target_vertices = None
    for candidate in reversed(baseline_vertices):
        subset = df[df["vertices"] == candidate]
        if all(
            not subset[(subset["mode"] == mode) & (subset["variant"] == variant)].empty
            for mode, variant in required_strategies
        ):
            target_vertices = candidate
            break

    if target_vertices is None:
        # Fallback to the largest baseline size even if some strategies are missing.
        target_vertices = int(baseline_vertices[-1])
        subset = df[df["vertices"] == target_vertices]
    else:
        subset = df[df["vertices"] == target_vertices]

    target_bytes = target_vertices * 64
    target_label = format_working_set(target_bytes)
    stats = []
    labels = {
        ("baseline", "standard"): "4 KiB pages",
        ("baseline", "hugetlb_2m"): "2 MiB pages",
        ("baseline", "hugetlb_1g"): "1 GiB pages",
    }

    blocked_rows = subset[subset["mode"] == "blocked"]
    if not blocked_rows.empty:
        labels[("blocked", "standard")] = "TLB blocking w/ 4 KiB pages"
    for (mode, variant), group in subset.groupby(["mode", "variant"]):
        label = labels.get((mode, variant))
        if not label:
            continue
        row = {"strategy": label, "mode": mode, "variant": variant}
        for q in quantiles:
            value = group["nanoseconds"].quantile(q)
            row[f"P{int(q*100):02d}"] = float(value)
        row["mean"] = float(group["nanoseconds"].mean())
        stats.append(row)

    stats_df = pd.DataFrame(stats)
    display_order = [
        "4 KiB pages",
        "2 MiB pages",
        "1 GiB pages",
        "TLB blocking w/ 4 KiB pages",
    ]
    present_order = [label for label in display_order if label in stats_df["strategy"].values]

    melt_df = stats_df.melt(
        id_vars=["strategy"],
        value_vars=[f"P{int(q*100):02d}" for q in quantiles],
        var_name="quantile",
        value_name="nanoseconds",
    )

    plt.figure(figsize=(7.5, 4.5))
    sns.barplot(
        data=melt_df,
        x="strategy",
        y="nanoseconds",
        hue="quantile",
        order=present_order or None,
    )
    plt.ylabel("Per-hop latency (ns)")
    plt.xlabel("")
    plt.title(f"Mitigations at {target_label} working set")
    plt.xticks(rotation=15, ha="right")
    plt.legend(title="")
    plt.tight_layout()
    out_path = FIG_DIR / "mitigation_bars.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return stats_df.set_index("strategy").to_dict(orient="index")


def load_perf_stats(run_df: pd.DataFrame) -> pd.DataFrame:
    perf_dir = DATA_DIR / "perf"
    if not perf_dir.exists():
        return pd.DataFrame()
    try:
        metadata = json.loads((DATA_DIR / "metadata.json").read_text())
        event_map: dict[str, str] = metadata.get("perf_events", {})
    except FileNotFoundError:
        event_map = {}
    reverse_event_map = {v: k for k, v in event_map.items()}
    records: list[dict[str, Any]] = []
    for path in sorted(perf_dir.glob("*_perf.csv")):
        if path.name.endswith("_warmup_perf.csv"):
            continue
        filename = path.name.replace("_perf.csv", ".csv")
        warmup_counts: dict[str, float] = {}
        warmup_path = perf_dir / path.name.replace("_perf.csv", "_warmup_perf.csv")
        if warmup_path.exists():
            with warmup_path.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(",")
                    if len(parts) < 3:
                        continue
                    try:
                        value = float(parts[0])
                    except ValueError:
                        continue
                    raw_event = parts[2]
                    warmup_counts[raw_event] = value
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) < 3:
                    continue
                try:
                    value = float(parts[0])
                except ValueError:
                    continue
                raw_event = parts[2]
                event = reverse_event_map.get(raw_event, raw_event)
                warmup_value = warmup_counts.get(raw_event, 0.0)
                adjusted = value - warmup_value
                if adjusted < 0:
                    adjusted = 0.0
                records.append({
                    "filename": filename,
                    "event": event,
                    "raw_event": raw_event,
                    "value": adjusted,
                    "raw_value": value,
                    "warmup_value": warmup_value,
                })
    if not records:
        return pd.DataFrame()
    perf_df = pd.DataFrame(records)
    perf_df = perf_df.merge(run_df, on="filename", how="left")
    perf_df["per_hop"] = perf_df["value"] / perf_df["steps"]
    perf_df["ns_per_cycle"] = perf_df.groupby("filename")["ns_per_cycle"].ffill()
    perf_df["ns_per_hop"] = perf_df["per_hop"] * perf_df["ns_per_cycle"]
    perf_df["working_set_mb"] = perf_df["working_set_bytes"] / (1024**2)
    return perf_df


def plot_translation_walks(perf_df: pd.DataFrame) -> None:
    if perf_df.empty:
        print("[perf] No perf metrics available for plotting")
        return
    subset = perf_df[(perf_df["mode"] == "baseline") & (perf_df["event"] == "walks")]
    if subset.empty:
        print("[perf] Walk events not available; skipping translation plot")
        return

    variant_labels = {
        "standard": "4 KiB pages",
        "hugetlb_2m": "2 MiB pages",
        "hugetlb_1g": "1 GiB pages",
    }
    markers = {
        "standard": "o",
        "hugetlb_2m": "s",
        "hugetlb_1g": "^",
    }

    plt.figure(figsize=(9, 5.5))
    all_ws = sorted(subset["working_set_bytes"].unique())
    xtick_vals = [ws / (1024**2) for ws in all_ws]
    drew_line = False
    for variant, label in variant_labels.items():
        variant_df = subset[subset["variant"] == variant]
        if variant_df.empty:
            continue
        grouped = (
            variant_df.groupby("working_set_bytes")["per_hop"].first().sort_index()
        )
        x = [ws / (1024**2) for ws in grouped.index]
        if not x:
            continue
        plt.plot(
            x,
            grouped.values,
            marker=markers.get(variant, "o"),
            label=label,
        )
        drew_line = True

    if not drew_line:
        plt.close()
        print("[perf] No translation walk data available for plotting")
        return
    plt.xscale("log", base=2)
    plt.yscale("log")
    xticklabels = [format_working_set(int(ws)) for ws in all_ws]
    plt.xticks(xtick_vals, xticklabels, rotation=30, ha="right")
    plt.xlabel("Working set size")
    plt.ylabel("Walks per hop")
    plt.title("Translation walks per hop vs. working set size")
    plt.grid(True, which="both", linestyle=":", linewidth=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "translation_walks.png", dpi=200)
    plt.close()


def plot_tlb_misses(perf_df: pd.DataFrame) -> None:
    if perf_df.empty:
        return
    target_event = "dTLB-load-misses"
    subset = perf_df[
        (perf_df["mode"] == "baseline") & (perf_df["raw_event"] == target_event)
    ]
    if subset.empty:
        print("[perf] No TLB miss cycle data available; skipping TLB miss plot")
        return

    variant_labels = {
        "standard": "4 KiB pages",
        "hugetlb_2m": "2 MiB pages",
        "hugetlb_1g": "1 GiB pages",
    }
    markers = {
        "standard": "o",
        "hugetlb_2m": "s",
        "hugetlb_1g": "^",
    }

    plt.figure(figsize=(9, 5.5))
    all_ws = sorted(subset["working_set_bytes"].unique())
    xtick_vals = [ws / (1024**2) for ws in all_ws]
    drew_line = False

    for variant, label in variant_labels.items():
        variant_df = subset[subset["variant"] == variant]
        if variant_df.empty:
            continue
        grouped = (
            variant_df.groupby("working_set_bytes")["per_hop"].first().sort_index()
        )
        x = [ws / (1024**2) for ws in grouped.index]
        if not x:
            continue
        plt.plot(
            x,
            grouped.values,
            marker=markers.get(variant, "o"),
            label=label,
        )
        drew_line = True

    if not drew_line:
        plt.close()
        print("[perf] No TLB miss cycles recorded; skipping plot")
        return

    plt.xscale("log", base=2)
    plt.yscale("log")
    xticklabels = [format_working_set(int(ws)) for ws in all_ws]
    plt.xticks(xtick_vals, xticklabels, rotation=30, ha="right")
    plt.xlabel("Working set size")
    plt.ylabel(f"{target_event} per hop")
    plt.title(f"{target_event} per hop vs. working set size")
    plt.grid(True, which="both", linestyle=":", linewidth=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "tlb_misses.png", dpi=200)
    plt.close()


def plot_llc_misses(perf_df: pd.DataFrame) -> None:
    if perf_df.empty:
        return
    target_event = "mem_load_retired.l3_miss"
    subset = perf_df[
        (perf_df["mode"] == "baseline") & (perf_df["raw_event"] == target_event)
    ]
    if subset.empty:
        print("[perf] No LLC miss event data available; skipping LLC plot")
        return

    variant_labels = {
        "standard": "4 KiB pages",
        "hugetlb_2m": "2 MiB pages",
        "hugetlb_1g": "1 GiB pages",
    }
    markers = {
        "standard": "o",
        "hugetlb_2m": "s",
        "hugetlb_1g": "^",
    }

    plt.figure(figsize=(9, 5.5))
    all_ws = sorted(subset["working_set_bytes"].unique())
    xtick_vals = [ws / (1024**2) for ws in all_ws]
    drew_line = False

    for variant, label in variant_labels.items():
        variant_df = subset[subset["variant"] == variant]
        if variant_df.empty:
            continue
        grouped = (
            variant_df.groupby("working_set_bytes")["per_hop"].first().sort_index()
        )
        x = [ws / (1024**2) for ws in grouped.index]
        if not x:
            continue
        plt.plot(
            x,
            grouped.values,
            marker=markers.get(variant, "o"),
            label=label,
        )
        drew_line = True

    if not drew_line:
        plt.close()
        print("[perf] No LLC miss data plotted; skipping figure")
        return

    plt.xscale("log", base=2)
    plt.yscale("log")
    xticklabels = [format_working_set(int(ws)) for ws in all_ws]
    plt.xticks(xtick_vals, xticklabels, rotation=30, ha="right")
    plt.xlabel("Working set size")
    plt.ylabel(f"{target_event} per hop")
    plt.title(f"{target_event} per hop vs. working set size")
    plt.grid(True, which="both", linestyle=":", linewidth=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "llc_misses.png", dpi=200)
    plt.close()


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df, run_df = load_dataset()
    perf_df = load_perf_stats(run_df)
    general_quantiles = [0.5, 0.95]
    quantile_labels = {0.5: "P50", 0.95: "P95"}
    q_df = compute_quantiles(df, general_quantiles, quantile_labels)
    plot_baseline_standard_quantiles(df)
    plot_baseline_quantiles(q_df)
    plot_baseline_2m_quantiles(df)
    plot_tlb_misses(perf_df)
    mitigation_stats = plot_mitigation_bars(df, general_quantiles)

    summary = {
        "quantiles": q_df.to_dict(orient="records"),
        "mitigation": mitigation_stats,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    main()
