"""Recompute the revision evidence from archived results and training logs.

No training. Reads the historical study tree (``--root``) and reports:

  1. Item-only minus full RLMRec on Amazon under three protocols.
  2. Our five-seed means against RLMRec's published Table 1 (LightGCN backbone).
  3. A replay of the patience-5 rule on the patience-25 Steam trajectories.
  4. When BPR leaves its ln 2 plateau in the Yelp alpha=0.03 diagnostics.

With ``--out-dir`` it also writes revision_evidence.json, the two LaTeX row
files and patience_bpr_curves.png used by the manuscript.

    python experiments/revision_evidence.py --root "<study tree>" \
        --out-dir papers/generated
"""

import argparse
import glob
import itertools
import json
import math
import os
import re
import statistics

from scipy import stats

SEEDS = [1, 7, 13, 42, 2024]
# Ren et al., WWW 2024, Table 1, Recall@20 / NDCG@20.
PUBLISHED = {
    "amazon": {"lightgcn": (0.1411, 0.0856), "full": (0.1483, 0.0903)},
    "yelp": {"lightgcn": (0.1157, 0.0733), "full": (0.1230, 0.0776)},
    "steam": {"lightgcn": (0.1348, 0.0855), "full": (0.1421, 0.0902)},
}
VAL = re.compile(r"Epoch\s+(\d+) Validation set \[.*?recall@20: ([\d.]+)")
BPR = re.compile(r"\[Epoch\s+(\d+)\] bpr_loss: ([\d.]+)")


def load(path):
    d = json.load(open(path))
    d = d[0] if isinstance(d, list) else d
    return d["seed"], d["test"]["recall"][2], d["test"]["ndcg"][2], d["best_epoch"]


def runs(root, pattern):
    out = {}
    for p in glob.glob(os.path.join(root, "experiments/amazon_study/results", pattern)):
        seed, r, n, e = load(p)
        out[seed] = (r, n, e)
    return out


def paired(a, b):
    """Mean of a-b, 95% t interval, paired t p-value, exact sign-flip p-value."""
    seeds = sorted(set(a) & set(b))
    d = [a[s] - b[s] for s in seeds]
    n, mu = len(d), statistics.mean(d)
    se = statistics.stdev(d) / math.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    p_t = 2 * stats.t.sf(abs(mu / se), n - 1)
    flips = [abs(sum(s * x for s, x in zip(sg, d)))
             for sg in itertools.product([1, -1], repeat=n)]
    p_s = sum(f >= abs(sum(d)) - 1e-15 for f in flips) / len(flips)
    return {"seeds": seeds, "diff": mu, "ci": [mu - h, mu + h], "p_t": p_t, "p_sign": p_s}


def metric(rs, i):
    return {s: v[i] for s, v in rs.items()}


def amazon_protocols(root):
    b4096_full = runs(root, "existing_published32/amazon_LightGCN_plus_RLMRec_Exact_seed*_exact.json")
    b4096_full.update(runs(root, "profile32/amazon_LightGCN_plus_RLMRec_Exact_published32_seed*.json"))
    b4096_lgcn = runs(root, "existing_published32/amazon_LightGCN_RLMRec_Exact_seed*_exact.json")
    b4096_lgcn.update(runs(root, "profile32/amazon_LightGCN_RLMRec_Exact_published32_seed*.json"))
    b4096_item = runs(root, "profile32/amazon_RLMRec_ItemOnly_32_published32_seed*.json")
    protocols = {
        "32D_b4096": (b4096_lgcn, b4096_full, b4096_item),
        "32D_b1024": (runs(root, "controlled32/*_lightgcn_seed*.json"),
                      runs(root, "controlled32/*_rlmrec_full_seed*.json"),
                      runs(root, "controlled32/*_rlmrec_item_only_seed*.json")),
        "64D_b1024": (runs(root, "existing_capacity64/*LightGCN_RLMRec_64*.json"),
                      runs(root, "existing_capacity64/*LightGCN_plus_RLMRec_64*.json"),
                      runs(root, "profile64/*RLMRec_ItemOnly_64*.json")),
    }
    out = {}
    for name, (lgcn, full, item) in protocols.items():
        out[name] = {
            "means": {m: [statistics.mean(metric(rs, i).values()) for i in (0, 1)]
                      for m, rs in (("lightgcn", lgcn), ("full", full), ("item", item))},
            "item_minus_full": {k: paired(metric(item, i), metric(full, i))
                                for i, k in ((0, "recall20"), (1, "ndcg20"))},
            "full_minus_lightgcn_recall20": paired(metric(full, 0), metric(lgcn, 0)),
        }
    return out


def published(root):
    sets = {
        "amazon": ("32D_b4096", None),
        "yelp": ("yelp32_baseline/*_lightgcn_seed*.json",
                 "yelp32_baseline/*_rlmrec_con_seed*.json"),
        "steam": ("steam32_confirmation/*confirm_lightgcn*.json",
                  "steam32_confirmation/*confirm_rlmrec_full*.json"),
    }
    out = {}
    for ds, (lg, fu) in sets.items():
        if fu is None:
            means = amazon_protocols(root)[lg]["means"]
            ours = {"lightgcn": means["lightgcn"], "full": means["full"]}
        else:
            ours = {m: [statistics.mean(metric(runs(root, g), i).values()) for i in (0, 1)]
                    for m, g in (("lightgcn", lg), ("full", fu))}
        out[ds] = {m: {"ours": ours[m], "published": PUBLISHED[ds][m],
                       "delta": [ours[m][i] - PUBLISHED[ds][m][i] for i in (0, 1)]}
                   for m in ("lightgcn", "full")}
    return out


def patience_replay(root):
    """Longest non-improving run before the recorded best epoch, counting
    rounded ties as non-improving. Below 5 means patience 5 keeps the same
    checkpoint, because patience only truncates an otherwise identical run."""
    out = {}
    for model, tag in (("LightGCN_RLMRec_Exact", "lightgcn"),
                       ("LightGCN_plus_RLMRec_Exact", "rlmrec_full"),
                       ("RLMRec_ItemOnly_32", "item_only")):
        rows = []
        for seed in SEEDS:
            log, = glob.glob(os.path.join(root, f"output/log/{model}/steam_cmp_seed{seed}_*.log"))
            traj = [(int(e), float(r)) for e, r in VAL.findall(open(log).read())]
            _, _, _, best = load(os.path.join(
                root, f"experiments/amazon_study/results/steam32_confirmation/"
                      f"steam_steam32_confirm_{tag}_seed{seed}.json"))
            top, run, longest = -1.0, 0, 0
            for e, r in traj:
                if e > best:
                    break
                run = 0 if (r > top or e == best) else run + 1
                longest, top = max(longest, run), max(top, r)
            rows.append({"seed": seed, "best_epoch": best, "longest_nonimproving": longest})
        out[tag] = {"runs": rows, "same_checkpoint_at_patience5":
                    all(r["longest_nonimproving"] < 5 for r in rows)}
    return out


def bpr_plateau(root, threshold=0.65):
    out = {}
    for log in sorted(glob.glob(os.path.join(
            root, "experiments/publication/results/yelp_patience_logs/*.log"))):
        text = open(log).read()
        bpr = [(int(e), float(v)) for e, v in BPR.findall(text)]
        val = [(int(e), float(r)) for e, r in VAL.findall(text)]
        out[os.path.basename(log)] = {
            "first_epoch_bpr_below": next(e for e, v in bpr if v < threshold),
            "epoch0_val_recall20": val[0][1],
            "min_val_recall20_first_15_epochs": min(r for e, r in val if e <= 15),
        }
    return out


def write_tables(result, out_dir):
    names = {"amazon": "Amazon", "yelp": "Yelp", "steam": "Steam"}
    rows = []
    for ds, v in result["published_comparison"].items():
        for m, label in (("lightgcn", "LightGCN"), ("full", "RLMRec-Con")):
            x = v[m]
            rows.append(f"{names[ds]} & {label} & {x['published'][0]:.4f} & {x['ours'][0]:.4f}"
                        f" & ${x['delta'][0]:+.4f}$ & {x['published'][1]:.4f}"
                        f" & {x['ours'][1]:.4f} & ${x['delta'][1]:+.4f}$ \\\\")
    with open(os.path.join(out_dir, "published_rows.tex"), "w") as f:
        f.write("\\newcommand{\\PublicationPublishedRows}{%\n" + "\n".join(rows) + "\n}\n")
    labels = {"32D_b4096": "32 & 4096", "32D_b1024": "32 & 1024", "64D_b1024": "64 & 1024"}
    rows = []
    for p, v in result["amazon_protocols"].items():
        r, n = v["item_minus_full"]["recall20"], v["item_minus_full"]["ndcg20"]
        rows.append(f"{labels[p]} & {v['means']['full'][0]:.4f} & {v['means']['item'][0]:.4f}"
                    f" & ${r['diff']:+.5f}$ & $[{r['ci'][0]:+.5f}, {r['ci'][1]:+.5f}]$"
                    f" & ${n['diff']:+.5f}$ & {n['p_t']:.3f} \\\\")
    with open(os.path.join(out_dir, "protocol_rows.tex"), "w") as f:
        f.write("\\newcommand{\\PublicationProtocolRows}{%\n" + "\n".join(rows) + "\n}\n")


def plot_patience(root, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(6.5, 3.4), sharex="col", sharey="row",
                            constrained_layout=True)
    for col, seed in enumerate((2023, 2025)):
        for model, label, style in (("LightGCN_plus_RLMRec_Exact", "Full", "-"),
                                    ("RLMRec_ItemOnly_32", "Item-only", "--")):
            text = open(os.path.join(
                root, "experiments/publication/results/yelp_patience_logs",
                f"yelp_{model}_alpha0.03_patience20_seed{seed}.log")).read()
            for row, pattern in ((0, VAL), (1, BPR)):
                x, y = zip(*[(int(e), float(v)) for e, v in pattern.findall(text)])
                axs[row, col].plot(x, y, style, label=label, linewidth=1.4)
        for row in (0, 1):
            axs[row, col].axvline(15, color="gray", linewidth=0.8, linestyle=":")
            axs[row, col].tick_params(labelsize=8)
        axs[1, col].axhline(math.log(2), color="gray", linewidth=0.8, linestyle="-.")
        axs[0, col].set_title(f"Seed {seed}", fontsize=9)
        axs[1, col].set_xlabel("Epoch index", fontsize=8)
    axs[0, 0].set_ylabel("Validation Recall@20", fontsize=8)
    axs[1, 0].set_ylabel("Training BPR loss", fontsize=8)
    axs[0, 1].legend(fontsize=8, loc="lower right")
    fig.savefig(os.path.join(out_dir, "patience_bpr_curves.png"), dpi=220)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-dir")
    args = ap.parse_args()
    result = {
        "amazon_protocols": amazon_protocols(args.root),
        "published_comparison": published(args.root),
        "steam_patience_replay": patience_replay(args.root),
        "yelp_bpr_plateau": bpr_plateau(args.root),
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.out_dir:
        with open(os.path.join(args.out_dir, "revision_evidence.json"), "w") as f:
            f.write(text + "\n")
        write_tables(result, args.out_dir)
        plot_patience(args.root, args.out_dir)


if __name__ == "__main__":
    main()
