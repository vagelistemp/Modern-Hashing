"""Charts from the benchmark CSVs (results/*.csv) -> results/figures/*.png.

Usage:  python scripts/plot.py [results_dir]
Only needs matplotlib. Implementation names are the identifiers used in the code.
"""
import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

for f in ("cambria.ttc", "cambriab.ttf", "cambriai.ttf", "calibri.ttf", "calibrib.ttf"):
    p = os.path.join("C:/Windows/Fonts", f)
    if os.path.exists(p):
        fm.fontManager.addfont(p)
plt.rcParams.update({
    "font.family": ["Cambria", "Calibri", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#dde3ea",
    "grid.linewidth": 0.7,
    "axes.edgecolor": "#8a96a3",
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.2,
    "figure.dpi": 130,
})

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

# one colour and one marker per implementation, the same in every figure
STYLE = {
    "slow-partition": ("#D1495B", "o", "-"),
    "indexed[btree-packed7]": ("#2B6CB0", "s", "-"),
    "indexed[std-btree] (thesis-like)": ("#7AA6D6", "^", "--"),
    "indexed[std-hash]": ("#00798C", "D", "--"),
    "k-tries[btree-packed7]": ("#E09F1F", "v", "-"),
    "std HashMap (SipHash)": ("#8D99AE", "x", ":"),
    "std HashMap (Fx hasher)": ("#2D3142", "+", ":"),
}
PHASE_GR = {
    "insert_new": "insert of a new key",
    "upsert": "insert or update (upsert)",
    "get_hit": "query: key present",
    "get_miss": "query: key absent",
    "remove": "delete",
}


def read(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        print("missing", path)
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def style(impl):
    return STYLE.get(impl, ("#555555", "o", "-"))


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", p)


# ------------------------------------------------------------------------------------------------ e1
def fig_e1():
    rows = read("e1_bucket_size.csv")
    if not rows:
        return
    phases = ["get_hit", "get_miss", "remove", "insert_new"]
    data = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        data[r["phase"]][r["impl"]][int(r["mean_bucket_load"])] = float(r["ns_per_op_median"])
    n = rows[0]["n"]
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.5))
    for ax, ph in zip(axes, phases):
        for impl, pts in data[ph].items():
            c, m, ls = style(impl)
            xs = sorted(pts)
            ax.plot(xs, [pts[x] for x in xs], color=c, marker=m, ls=ls, lw=1.6, ms=4.5, label=impl)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(PHASE_GR[ph])
        ax.set_xlabel("mean number of keys per bucket")
    axes[0].set_ylabel("ns per operation (median)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.12))
    n_gr = f"{int(n):,}"
    fig.suptitle(f"Cost per operation as the bucket grows (n = {n_gr} keys, random keys)", y=1.03, fontsize=11)
    save(fig, "e1_bucket_size.png")


# ------------------------------------------------------------------------------------------------ e2
def fig_e2():
    rows = read("e2_scaling.csv")
    if not rows:
        return
    for mode in ("random", "sequential"):
        for lf in ("0.9", "0.5"):
            sel = [r for r in rows if r["keys"] == mode and r["lf"] == lf]
            if not sel:
                continue
            data = defaultdict(lambda: defaultdict(dict))
            for r in sel:
                data[r["phase"]][r["impl"]][int(r["n"])] = float(r["ns_per_op_median"])
            fig, axes = plt.subplots(1, 5, figsize=(16.5, 3.4))
            for ax, ph in zip(axes, ["insert_new", "upsert", "get_hit", "get_miss", "remove"]):
                for impl, pts in data[ph].items():
                    c, m, ls = style(impl)
                    xs = sorted(pts)
                    ax.plot(xs, [pts[x] for x in xs], color=c, marker=m, ls=ls, lw=1.5, ms=4.2, label=impl)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_title(PHASE_GR[ph])
                ax.set_xlabel("capacity n")
            axes[0].set_ylabel("ns per operation (median)")
            h, l = axes[0].get_legend_handles_labels()
            fig.legend(h, l, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.12))
            fig.suptitle(
                f"Scaling with n: load factor {lf}, {'random' if mode == 'random' else 'sequential'} keys",
                y=1.03, fontsize=11)
            save(fig, f"e2_scaling_{mode}_lf{lf.replace('.', '')}.png")

    # effect of the load factor at the largest size common to all implementations
    sel = [r for r in rows if r["keys"] == "random"]
    if sel:
        ns = sorted({int(r["n"]) for r in sel})
        n = ns[-1] if all(any(int(r["n"]) == ns[-1] and r["impl"] == i for r in sel) for i in STYLE) else ns[-2]
        fig, ax = plt.subplots(figsize=(7.4, 3.6))
        impls = [i for i in STYLE if any(r["impl"] == i and int(r["n"]) == n for r in sel)]
        width = 0.38
        for j, lf in enumerate(("0.5", "0.9")):
            vals = []
            for impl in impls:
                tot = 0.0
                for ph in ("insert_new", "get_hit", "remove"):
                    for r in sel:
                        if r["impl"] == impl and int(r["n"]) == n and r["lf"] == lf and r["phase"] == ph:
                            tot += float(r["ns_per_op_median"])
                vals.append(tot)
            ax.bar([x + (j - 0.5) * width for x in range(len(impls))], vals, width, label=f"load factor {lf}",
                   color=["#9fb8d6", "#2B6CB0"][j])
        ax.set_xticks(range(len(impls)))
        ax.set_xticklabels([i.replace(" (", "\n(").replace("[", "\n[") for i in impls], fontsize=7)
        ax.set_yscale("log")
        ax.set_ylabel("ns: insert + query + delete")
        ax.set_title(f"Effect of the load factor (n = {n:,}, random keys)")
        ax.legend(frameon=False)
        save(fig, "e2_load_factor.png")


# ------------------------------------------------------------------------------------------------ e3
def fig_e3():
    rows = read("e3_latency.csv")
    if not rows:
        return
    cfgs = []
    for r in rows:
        if r["config"] not in cfgs:
            cfgs.append(r["config"])
    cfg_gr = {"paper-geometry": "paper geometry", "14-bit-fingerprints": "14-bit fingerprints (many rebuilds)"}
    fig, axes = plt.subplots(len(cfgs), 2, figsize=(11.2, 3.4 * len(cfgs)), squeeze=False)
    cols = [("p50_ns", "p50"), ("p99_ns", "p99"), ("p999_ns", "p99.9"), ("p9999_ns", "p99.99"), ("max_ns", "max")]
    impls = ["indexed[btree-packed7]", "k-tries[btree-packed7]", "std HashMap (Fx hasher)"]
    for i, cfg in enumerate(cfgs):
        for j, op in enumerate(("insert", "delete")):
            ax = axes[i][j]
            width = 0.26
            for k, impl in enumerate(impls):
                r = [x for x in rows if x["config"] == cfg and x["impl"] == impl and x["operation"] == op]
                if not r:
                    continue
                r = r[0]
                vals = [max(float(r[c]), 50.0) for c, _ in cols]
                ax.bar([x + (k - 1) * width for x in range(len(cols))], vals, width, color=style(impl)[0], label=impl)
            ax.set_yscale("log")
            ax.set_xticks(range(len(cols)))
            ax.set_xticklabels([n for _, n in cols])
            ax.set_title(f"{'insert' if op == 'insert' else 'delete'}: {cfg_gr.get(cfg, cfg)}")
            ax.set_ylabel("ns (timer resolution ~100 ns)")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Latency distribution of individual operations", y=1.0, fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    save(fig, "e3_latency.png")


# ------------------------------------------------------------------------------------------------ e4
def fig_e4():
    rows = read("e4_space.csv")
    if not rows:
        return
    impls = ["slow-partition", "indexed[btree-packed7]", "indexed[sorted-vec]", "k-tries[btree-packed7]"]
    ns = sorted({int(r["n"]) for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.6))
    ax = axes[0]
    width = 0.2
    for k, impl in enumerate(impls):
        vals = []
        for n in ns:
            r = [x for x in rows if x["impl"] == impl and int(x["n"]) == n]
            vals.append(float(r[0]["overhead_bits_per_item"]) if r else 0.0)
        ax.bar([x + (k - 1.5) * width for x in range(len(ns))], vals, width, label=impl, color=style(impl)[0] if impl in STYLE else "#6a994e")
    ref = [float(next(x for x in rows if int(x["n"]) == n)["paper_loglogn"]) for n in ns]
    ax.plot(range(len(ns)), ref, color="black", marker="o", ms=3, lw=1, ls="--", label="log₂ log₂ n (the paper's order)")
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels([f"{n:,}" for n in ns], fontsize=7.5)
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("bits per key above w = 32")
    ax.set_title("Extra space per key")
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    parts = [("key_slot_bits_per_item", "key slots (32 bits + slack)", "#9fb8d6"),
             ("mapper_bits_per_item", "query mappers", "#2B6CB0"),
             ("buffer_bits_per_item", "buffers (K-Tries)", "#E09F1F")]
    last = max(ns)
    sel = [i for i in impls if any(x["impl"] == i and int(x["n"]) == last for x in rows)]
    bottoms = [0.0] * len(sel)
    for key, label, color in parts:
        vals = [float(next(x for x in rows if x["impl"] == i and int(x["n"]) == last)[key]) for i in sel]
        ax.bar(range(len(sel)), vals, 0.55, bottom=bottoms, label=label, color=color)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.axhline(32, color="black", lw=0.9, ls="--")
    ax.text(len(sel) - 0.5, 33, "w = 32", ha="right", fontsize=8)
    ax.set_xticks(range(len(sel)))
    ax.set_xticklabels([s.replace("[", "\n[") for s in sel], fontsize=7.5)
    ax.set_ylabel("bits per key")
    ax.set_title(f"Where the bits go (n = {last:,})")
    ax.legend(frameon=False, fontsize=7.5)
    save(fig, "e4_space.png")


# ------------------------------------------------------------------------------------------------ e5
def fig_e5():
    rows = [r for r in read("e5_four_russians.csv") if r["experiment"] == "e5"]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    names = [r["mapper"].replace(" (every node op is a table read)", "\n(every node op is a table read)") for r in rows]
    vals = [float(r["ns_per_op"]) for r in rows]
    ax.barh(range(len(rows)), vals, color=["#E09F1F" if "tabulated" in n else "#2B6CB0" for n in names])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.0f} ns", va="center", fontsize=8)
    ax.set_xlabel("ns per operation (universe of 16 fingerprints)")
    ax.set_title("Query mapper with tiny parameters: precomputed tables vs. computation")
    save(fig, "e5_four_russians.png")


# ------------------------------------------------------------------------------------------------ e6
def fig_e6():
    a = read("e6a_slab_space.csv")
    b = read("e6b_resizable.csv")
    if not a and not b:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.6))
    ax = axes[0]
    if a:
        fills = sorted({float(r["fill"]) for r in a})
        width = 0.36
        for j, (label, color) in enumerate([("indexed (fixed buckets)", "#9fb8d6"), ("indexed (slab buckets)", "#2B6CB0")]):
            vals = []
            for f in fills:
                r = [x for x in a if x["impl"] == label and float(x["fill"]) == f]
                vals.append(float(r[0]["key_slot_bits_per_item"]) if r else 0.0)
            ax.bar([i + (j - 0.5) * width for i in range(len(fills))], vals, width, color=color,
                   label="fixed-size slots" if j == 0 else "slab slots (§6)")
        ax.axhline(32, color="black", lw=0.9, ls="--")
        ax.text(len(fills) - 0.6, 36, "w = 32", ha="right", fontsize=8)
        ax.set_xticks(range(len(fills)))
        ax.set_xticklabels([f"{f:.0%}" for f in fills])
        ax.set_yscale("log")
        ax.set_xlabel("how full the table is (relative to capacity n)")
        ax.set_ylabel("key-slot bits per stored key")
        ax.set_title("Memory follows the keys, not the capacity")
        ax.legend(frameon=False)
    ax = axes[1]
    if b:
        xs, caps, lens, bits = [], [], [], []
        off = 0
        for r in b:
            ops = int(r["ops_done"]) + (0 if r["phase"] == "grow" else max(int(x["ops_done"]) for x in b if x["phase"] == "grow"))
            xs.append(ops)
            caps.append(int(r["capacity"]))
            lens.append(int(r["len"]))
            # bits per key is meaningless for an empty table: leave a gap
            bits.append(float(r["bits_per_key"]) if int(r["len"]) > 0 else float("nan"))
        ax.plot(xs, caps, color="#E09F1F", marker="o", ms=3.5, label="table capacity")
        ax.plot(xs, lens, color="#2B6CB0", marker="s", ms=3.5, label="keys")
        ax.set_yscale("log")
        ax.set_xlabel("operations (inserts first, then deletes)")
        ax.set_ylabel("keys / capacity")
        ax2 = ax.twinx()
        ax2.plot(xs, bits, color="#D1495B", marker="^", ms=3.5, ls="--", label="bits per key (slots + mappers)")
        ax2.set_ylabel("bits per key")
        ax2.grid(False)
        ax2.spines["right"].set_visible(True)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7.5, loc="lower center")
        ax.set_title("Resizable: the table grows and shrinks")
    save(fig, "e6_resizing.png")


if __name__ == "__main__":
    fig_e1()
    fig_e2()
    fig_e3()
    fig_e4()
    fig_e5()
    fig_e6()
