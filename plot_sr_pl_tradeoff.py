import argparse
import json
import os
import re

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


OURS_DATA = [
    ["Ours", "Non-aware", 0.0, 97.20, 2.80, 16.59, 23.53],
    ["Ours", "Non-aware", 0.2, 93.84, 6.16, 15.01, 21.82],
    ["Ours", "Non-aware", 0.4, 88.16, 11.84, 13.90, 20.47],
    ["Ours", "Non-aware", 0.6, 83.12, 16.88, 13.25, 19.62],
    ["Ours", "Non-aware", 0.8, 77.36, 22.64, 12.66, 18.72],
    ["Ours", "Non-aware", 1.0, 73.12, 26.88, 12.35, 18.14],
    ["Ours", "Non-aware", "Adaptive", 97.04, 2.96, 16.51, 23.44],
    ["Ours", "Aware", 0.0, 100.00, 0.00, 16.24, 23.49],
    ["Ours", "Aware", 0.2, 99.92, 0.08, 14.34, 21.56],
    ["Ours", "Aware", 0.4, 99.92, 0.08, 12.61, 19.84],
    ["Ours", "Aware", 0.6, 99.68, 0.32, 11.73, 18.96],
    ["Ours", "Aware", 0.8, 99.42, 0.48, 11.16, 18.37],
    ["Ours", "Aware", 1.0, 99.20, 0.80, 10.87, 18.08],
    ["Ours", "Aware", "Adaptive", 99.28, 0.72, 10.94, 18.16],
    ["Ours", "Mixed", 0.0, 98.56, 1.36, 16.61, 23.79],
    ["Ours", "Mixed", 0.2, 98.00, 2.00, 14.88, 22.03],
    ["Ours", "Mixed", 0.4, 96.24, 3.76, 13.65, 20.66],
    ["Ours", "Mixed", 0.6, 94.16, 5.84, 12.66, 19.61],
    ["Ours", "Mixed", 0.8, 89.60, 10.40, 12.19, 18.90],
    ["Ours", "Mixed", 1.0, 86.08, 13.92, 11.85, 18.44],
    ["Ours", "Mixed", "Adaptive", 95.52, 4.48, 13.63, 20.64],
    ["Ours", "Spatial Clusters", 0.0, 97.12, 2.88, 15.47, 22.40],
    ["Ours", "Spatial Clusters", 0.2, 96.00, 4.00, 13.76, 20.73],
    ["Ours", "Spatial Clusters", 0.4, 94.64, 5.36, 12.84, 19.82],
    ["Ours", "Spatial Clusters", 0.6, 92.32, 7.68, 12.18, 19.13],
    ["Ours", "Spatial Clusters", 0.8, 90.64, 9.36, 11.87, 18.74],
    ["Ours", "Spatial Clusters", 1.0, 88.80, 11.20, 11.61, 18.37],
    ["Ours", "Spatial Clusters", "Adaptive", 95.36, 4.64, 12.85, 19.83],
]

SCENARIOS = {
    "seperate_all_ignorant": "Non-aware",
    "seperate_all_aware": "Aware",
    "seperate_mixed_5050": "Mixed",
    "cluster_aware_ignorant": "Spatial Clusters",
}
SCENARIO_ORDER = list(SCENARIOS.values())
SCALES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
METHOD_ORDER = ["Ours", "Full-finetune", "Action-space", "MPC"]


def _timestamp(entry):
    return entry.get("timestamp", "")


def _summary_to_row(method, scenario, scale, summaries):
    df = pd.DataFrame(summaries)
    return [
        method,
        scenario,
        scale,
        df["success_rate"].mean() * 100.0,
        df["collision_rate"].mean() * 100.0,
        df["avg_nav_time"].mean(),
        df["avg_path_length"].mean(),
    ]


def _load_aggregate(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _latest_seed_summaries(aggregate, prefix):
    latest_by_seed = {}
    for key, entry in aggregate.items():
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix):]
        seed = rest.split("_")[0]
        if seed not in latest_by_seed or _timestamp(entry) > _timestamp(latest_by_seed[seed]):
            latest_by_seed[seed] = entry
    return [entry["summary"] for entry in latest_by_seed.values()]


def _latest_adaptive_summaries(aggregate, prefix):
    latest_by_seed = {}
    for key, entry in aggregate.items():
        if not key.startswith(prefix):
            continue
        match = re.search(r"_exp([^_]+)", key)
        seed = match.group(1) if match else "default"
        if seed not in latest_by_seed or _timestamp(entry) > _timestamp(latest_by_seed[seed]):
            latest_by_seed[seed] = entry
    return [entry["summary"] for entry in latest_by_seed.values()]


def load_latest_sweep_data(aggregate_path, method, fixed_behaviour, exp_prefix, adaptive_behaviour=None):
    aggregate = _load_aggregate(aggregate_path)
    rows = []
    if not aggregate:
        return rows

    for scenario_key, scenario_label in SCENARIOS.items():
        for scale in SCALES:
            prefix = f"{scenario_key}_{fixed_behaviour}_exp{exp_prefix}_{scale}_"
            summaries = _latest_seed_summaries(aggregate, prefix)
            if summaries:
                rows.append(_summary_to_row(method, scenario_label, scale, summaries))

    if adaptive_behaviour:
        for scenario_key, scenario_label in SCENARIOS.items():
            prefix = f"{scenario_key}_{adaptive_behaviour}"
            summaries = _latest_adaptive_summaries(aggregate, prefix)
            if summaries:
                rows.append(_summary_to_row(method, scenario_label, "Adaptive", summaries))

    return rows


def load_latest_fullft_data(aggregate_path):
    return load_latest_sweep_data(
        aggregate_path,
        method="Full-finetune",
        fixed_behaviour="fixed_fullfinetune_scale",
        exp_prefix="fullft_scale",
        adaptive_behaviour="adaptive_fullfinetune_gt",
    )


def load_latest_action_data(aggregate_path):
    return load_latest_sweep_data(
        aggregate_path,
        method="Action-space",
        fixed_behaviour="fixed_action_scale",
        exp_prefix="action_scale",
        adaptive_behaviour="adaptive_action_gt",
    )


def load_latest_mpc_data(aggregate_path):
    return load_latest_sweep_data(
        aggregate_path,
        method="MPC",
        fixed_behaviour="mpc_fixed",
        exp_prefix="mpc_scale",
        adaptive_behaviour="mpc_adaptive",
    )


def _slugify(value):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _scale_sort_value(value):
    if value == "Adaptive":
        return len(SCALES)
    try:
        return SCALES.index(float(value))
    except (ValueError, TypeError):
        return len(SCALES) + 1


def _plot_one_tradeoff(plot_df, title, output_path, show):
    markers = {0.0: "o", 0.2: "v", 0.4: "s", 0.6: "P", 0.8: "^", 1.0: "D"}
    colors = {
        "Ours": "#1F77B4",
        "Full-finetune": "#D62728",
        "Action-space": "#2CA02C",
        "MPC": "#9467BD",
    }

    fig, ax = plt.subplots(figsize=(9.4, 4.8))

    for method in METHOD_ORDER:
        method_df = plot_df[plot_df["Method"] == method].copy()
        if method_df.empty:
            continue

        fixed_df = method_df[method_df["Scale"] != "Adaptive"].copy()
        if not fixed_df.empty:
            fixed_df["Scale"] = fixed_df["Scale"].astype(float)
            fixed_df = fixed_df.sort_values("Scale")
            ax.plot(fixed_df["PL"], fixed_df["SR"], linestyle="--",
                    linewidth=1.8, color=colors[method], alpha=0.45, zorder=1)

            for _, row in fixed_df.iterrows():
                scale = row["Scale"]
                ax.scatter(row["PL"], row["SR"], s=145, marker=markers[scale],
                           facecolor=colors[method], edgecolor="black", linewidth=1.1,
                           alpha=0.9, zorder=3)

        adaptive_rows = method_df[method_df["Scale"] == "Adaptive"]
        if len(adaptive_rows) > 0:
            adaptive = adaptive_rows.iloc[0]
            ax.scatter(adaptive["PL"], adaptive["SR"], s=280, marker="*",
                       facecolor=colors[method], edgecolor="black", linewidth=1.2,
                       zorder=4)

    ax.set_xlabel("Path Length (m)", fontsize=17)
    ax.set_ylabel("Success Rate (%)", fontsize=17)
    ax.set_title(title, fontsize=16)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, alpha=0.28)

    shape_handles = [
        Line2D([0], [0], marker=markers[scale], linestyle="None",
               markerfacecolor="white", markeredgecolor="black",
               markeredgewidth=1.1, markersize=9,
               label=f"Fixed scale = {scale:.1f}")
        for scale in SCALES
    ]
    shape_handles.append(
        Line2D([0], [0], marker="*", linestyle="None",
               markerfacecolor="white", markeredgecolor="black",
               markeredgewidth=1.1, markersize=13,
               label="Adaptive scale")
    )
    present_methods = set(plot_df["Method"])
    method_handles = [
        Line2D([0], [0], color=colors[method], marker="o", linestyle="-",
               linewidth=2, markersize=7, label=method)
        for method in METHOD_ORDER
        if method in present_methods
    ]
    ax.legend(handles=shape_handles + method_handles,
              loc="lower right", fontsize=8.2, frameon=True)

    plt.tight_layout()
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_tradeoffs(df, output_prefix, show):
    saved = []
    for scenario in SCENARIO_ORDER:
        scenario_df = df[df["Scenario"] == scenario].copy()
        if scenario_df.empty:
            continue

        scenario_df["_method_order"] = scenario_df["Method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
        scenario_df["_scale_order"] = scenario_df["Scale"].map(_scale_sort_value)
        scenario_df = scenario_df.sort_values(["_method_order", "_scale_order"]).drop(columns=["_method_order", "_scale_order"])

        print(f"\n{scenario}:")
        print(scenario_df[["Method", "Scale", "SR", "CR", "NT", "PL"]].round(3).to_string(index=False))
        output_path = f"{output_prefix}_{_slugify(scenario)}"
        _plot_one_tradeoff(scenario_df, scenario, output_path, show)
        saved.append(output_path)

    avg = (
        df.groupby(["Method", "Scale"], sort=False)[["SR", "CR", "NT", "PL"]]
        .mean()
        .reset_index()
    )
    avg["_method_order"] = avg["Method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    avg["_scale_order"] = avg["Scale"].map(_scale_sort_value)
    avg = avg.sort_values(["_method_order", "_scale_order"]).drop(columns=["_method_order", "_scale_order"])

    print("\nAverage across all scenarios:")
    print(avg.round(3).to_string(index=False))
    average_output = f"{output_prefix}_average"
    _plot_one_tradeoff(avg, "Average Across Scenarios", average_output, show)
    saved.append(average_output)

    _plot_one_tradeoff(avg, "Average Across Scenarios", output_prefix, False)
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", default="trained_models/LoraF_invi_visi_rank_1/test/all_evaluations.json",
                        help="Aggregate JSON containing LoRA-base rebuttal sweep results.")
    parser.add_argument("--fullft-aggregate", default=None,
                        help="Override aggregate JSON for full-finetune sweep results.")
    parser.add_argument("--action-aggregate", default=None,
                        help="Override aggregate JSON for action-space sweep results.")
    parser.add_argument("--mpc-aggregate", default=None,
                        help="Override aggregate JSON for MPC sweep results.")
    parser.add_argument("--output-prefix", default="sr_vs_pl_rebuttal_sweeps")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    aggregate = args.aggregate
    fullft_aggregate = args.fullft_aggregate or aggregate
    action_aggregate = args.action_aggregate or aggregate
    mpc_aggregate = args.mpc_aggregate or aggregate

    fullft_data = load_latest_fullft_data(fullft_aggregate)
    action_data = load_latest_action_data(action_aggregate)
    mpc_data = load_latest_mpc_data(mpc_aggregate)

    for name, rows, path in [
        ("full-finetune", fullft_data, fullft_aggregate),
        ("action-space", action_data, action_aggregate),
        ("MPC", mpc_data, mpc_aggregate),
    ]:
        if not rows:
            print(f"Warning: no {name} sweep entries found in {path}")

    df = pd.DataFrame(
        OURS_DATA + fullft_data + action_data + mpc_data,
        columns=["Method", "Scenario", "Scale", "SR", "CR", "NT", "PL"],
    )
    saved = plot_tradeoffs(df, args.output_prefix, args.show)
    print("\nSaved plots:")
    for path in saved:
        print(f"  {path}.png")


if __name__ == "__main__":
    main()
