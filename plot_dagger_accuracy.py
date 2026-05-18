"""Plot DAgger AwAcc convergence vs the open-loop "no distribution shift" reference.

For one (HUMAN_NUM, rollout_driver) combo (e.g. hn=20 + adaptive), draws per
scenario:

  - solid line: AwAcc under closed-loop deployment (the configured rollout
    behaviour), one point per DAgger iteration. Each iter's predictor was
    different — iter 0 = pre-DAgger, iter K = after K retrains.
  - horizontal dashed line: AwAcc under open-loop adaptive_gt (the predictor
    in shadow mode, with GT driving the policy). This is the "no distribution
    shift" reference — the predictor's true quality in-distribution.

The gap between the two is the closed-loop / DAgger gap. As DAgger runs, the
solid line should converge upward toward the dashed reference.

Usage (inside the gen_safe_py10 container):

    python3 plot_dagger_accuracy.py
    python3 plot_dagger_accuracy.py --human_num 30 --rollout_driver switching
    python3 plot_dagger_accuracy.py --human_num 20 --rollout_driver adaptive --out my_plot.png
"""

import argparse
import glob
import json
import os
from typing import Optional

import matplotlib.pyplot as plt


SCENARIOS = [
    "seperate_mixed_5050",
    "seperate_all_aware",
    "seperate_all_ignorant",
    "cluster_aware_ignorant",
]


def _summary_head(path: str, peek_bytes: int = 16384) -> Optional[dict]:
    """Read just enough to extract the summary block. Avoids parsing the
    multi-GB episodes array."""
    import re
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        head = fh.read(peek_bytes)
    re_int = lambda key: re.search(rb'"%s"\s*:\s*(\d+)' % key.encode(), head)
    re_float = lambda key: re.search(rb'"%s"\s*:\s*([-+0-9.eE]+)' % key.encode(), head)
    out = {}
    for k in ("tp", "tn", "fp", "fn", "num_episodes", "human_num"):
        m = re_int(k)
        if m:
            out[k] = int(m.group(1))
    for k in ("success_rate", "collision_rate", "avg_path_length",
              "avg_lora_scale", "avg_nav_time"):
        m = re_float(k)
        if m:
            try:
                out[k] = float(m.group(1))
            except ValueError:
                pass
    return out


def _aw_acc(s: Optional[dict]) -> Optional[float]:
    if not s:
        return None
    tp = s.get("tp", 0) or 0
    tn = s.get("tn", 0) or 0
    fp = s.get("fp", 0) or 0
    fn = s.get("fn", 0) or 0
    total = tp + tn + fp + fn
    return (tp + tn) / total if total > 0 else None


def _human_num_to_exp_suffix(hn: int) -> str:
    """Mapping the user's existing exp_id convention to HUMAN_NUM."""
    return {20: "", 30: "_exp1", 40: "_exp2"}.get(hn, "")


def _open_loop_paths(test_dir: str, hn: int):
    """Return per-scenario adaptive_gt dump path at this HUMAN_NUM (or None)."""
    suffix = _human_num_to_exp_suffix(hn)
    out = {}
    for sc in SCENARIOS:
        path = os.path.join(test_dir, f"{sc}_adaptive_gt{suffix}.json")
        out[sc] = path if os.path.exists(path) else None
    return out


def _dagger_iter_paths(test_dir: str, hn: int, driver: str):
    """For each scenario, return ordered list of (iter_label, path) for
    DAgger iter dumps under this (hn, driver) combo. Tries the new TAG'd
    naming first; falls back to the legacy expdagger{N} naming for the
    pre-TAG hn=20+adaptive sweep the user already ran."""
    rollout_behaviour = f"{driver}_pred"
    tag = f"h{hn}_{driver}"
    per_scenario = {}
    for sc in SCENARIOS:
        # 1) New tagged naming
        pattern = os.path.join(
            test_dir, f"{sc}_{rollout_behaviour}_expdagger_{tag}_iter*.json"
        )
        matches = sorted(glob.glob(pattern), key=_iter_index)
        if matches:
            per_scenario[sc] = [(f"iter{_iter_index(p)}", p) for p in matches]
            continue
        # 2) Legacy naming (pre-TAG): only for hn=20+adaptive
        if hn == 20 and driver == "adaptive":
            pattern = os.path.join(test_dir, f"{sc}_adaptive_pred_expdagger*.json")
            legacy = sorted(glob.glob(pattern), key=_legacy_iter_index)
            # filter out any new-tagged files accidentally matching
            legacy = [p for p in legacy if "_h" not in os.path.basename(p).split("expdagger")[-1]]
            per_scenario[sc] = [(f"iter{_legacy_iter_index(p)}", p) for p in legacy]
        else:
            per_scenario[sc] = []
    return per_scenario


def _iter_index(path: str) -> int:
    """Parse iter{N} from a tagged dump filename."""
    import re
    m = re.search(r"_iter(\d+)\.json$", path)
    return int(m.group(1)) if m else -1


def _legacy_iter_index(path: str) -> int:
    """Parse expdagger{N} from the pre-TAG dump filename."""
    import re
    m = re.search(r"_expdagger(\d+)\.json$", path)
    return int(m.group(1)) if m else -1


def _final_closed_loop_path(test_dir: str, hn: int, driver: str):
    """Path to the final post-DAgger closed-loop sweep dump per scenario.
    Tries hn-matching exp suffix; otherwise the unsuffixed file."""
    rollout_behaviour = f"{driver}_pred"
    suffix = _human_num_to_exp_suffix(hn)
    out = {}
    for sc in SCENARIOS:
        # Prefer hn-matched exp file (e.g. _exp1 for hn=30)
        candidate = os.path.join(test_dir, f"{sc}_{rollout_behaviour}{suffix}.json")
        if os.path.exists(candidate):
            out[sc] = candidate
            continue
        # Fall back to unsuffixed (hn=20 default)
        candidate = os.path.join(test_dir, f"{sc}_{rollout_behaviour}.json")
        out[sc] = candidate if os.path.exists(candidate) else None
    return out


def plot(model_dir: str, hn: int, driver: str, out_path: str):
    test_dir = os.path.join(model_dir, "test")

    open_loop = _open_loop_paths(test_dir, hn)
    iter_dumps = _dagger_iter_paths(test_dir, hn, driver)
    final_post = _final_closed_loop_path(test_dir, hn, driver)

    # Build per-scenario series: list of (label, aw_acc)
    series = {}
    for sc in SCENARIOS:
        pts = []
        # Iter 0 = first DAgger iter's rollout (before any DAgger retrain)
        for label, path in iter_dumps.get(sc, []):
            aw = _aw_acc(_summary_head(path))
            if aw is not None:
                pts.append((label, aw))
        # Final post-DAgger sweep, if present
        if final_post.get(sc):
            aw = _aw_acc(_summary_head(final_post[sc]))
            if aw is not None:
                # Number it as the next iter
                next_idx = (max(_iter_index_for_label(l) for l, _ in pts) + 1) if pts else 0
                pts.append((f"final", aw))
        series[sc] = pts

    open_loop_acc = {}
    for sc, path in open_loop.items():
        if path:
            aw = _aw_acc(_summary_head(path))
            if aw is not None:
                open_loop_acc[sc] = aw

    # Plot
    n = len(SCENARIOS)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]
    cmap = plt.get_cmap("tab10")

    for i, sc in enumerate(SCENARIOS):
        ax = axes[i]
        pts = series.get(sc, [])
        ref = open_loop_acc.get(sc)

        if pts:
            xs = list(range(len(pts)))
            ys = [a for _, a in pts]
            labels = [l for l, _ in pts]
            ax.plot(xs, ys, marker="o", color=cmap(0), linewidth=2,
                    label=f"closed-loop {driver}_pred")
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation=45 if max(len(l) for l in labels) > 4 else 0)
        else:
            ax.text(0.5, 0.5, "no DAgger dumps found",
                    ha="center", va="center", transform=ax.transAxes, color="gray")

        if ref is not None:
            ax.axhline(ref, linestyle="--", color="green", linewidth=1.8,
                       label=f"open-loop adaptive_gt  ({ref:.3f})")

        # Target line at 0.80
        ax.axhline(0.80, linestyle=":", color="red", alpha=0.5, linewidth=1.0,
                   label="target (0.80)")

        ax.set_ylim(0.0, 1.0)
        ax.set_title(sc, fontsize=11)
        if i == 0:
            ax.set_ylabel("AwAcc (predictor vs ground truth)")
        ax.set_xlabel("DAgger iteration")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle(
        f"FriendlyPredictor accuracy under DAgger  —  HUMAN_NUM={hn},  rollout={driver}_pred",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Saved plot → {out_path}")

    # Print the numbers we plotted so the user can reuse them in a table
    print()
    print("Per-scenario data (closed-loop AwAcc per iter // open-loop reference):")
    for sc in SCENARIOS:
        pts = series.get(sc, [])
        ref = open_loop_acc.get(sc)
        ref_str = f"{ref:.4f}" if ref is not None else "—"
        pts_str = "  ".join(f"{l}={a:.4f}" for l, a in pts) if pts else "(no iters)"
        print(f"  {sc:<36s}  open-loop={ref_str}    {pts_str}")


def _iter_index_for_label(label: str) -> int:
    import re
    m = re.search(r"\d+", label)
    return int(m.group(0)) if m else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_dir", default="trained_models/LoraF_invi_visi_rank_1")
    ap.add_argument("--human_num", type=int, default=20,
                    help="HUMAN_NUM that selects which dump set to plot (default 20).")
    ap.add_argument("--rollout_driver", choices=["adaptive", "switching"],
                    default="adaptive",
                    help="Which closed-loop driver was used during DAgger rollouts "
                         "(default adaptive → reads adaptive_pred files).")
    ap.add_argument("--out", default=None,
                    help="Output PNG path. Defaults to "
                         "dagger_awacc_h{HUMAN_NUM}_{driver}.png in the cwd.")
    args = ap.parse_args()

    out = args.out or f"dagger_awacc_h{args.human_num}_{args.rollout_driver}.png"
    plot(args.model_dir, args.human_num, args.rollout_driver, out)


if __name__ == "__main__":
    main()
