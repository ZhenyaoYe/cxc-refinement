import io
import zipfile
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from greedy_peeling import greedy_peeling_X_all


# =========================================================
# Options
# =========================================================
@dataclass
class CXCOptions:
    lambda_stage1: float = 1.80
    lambda_stage2: float = 1.70

    min_cells: int = 30
    min_block_size: int = 5

    corr_threshold: float = 0.05
    stage2_skip_mean_abs: float = 0.02

    parallel: bool = False
    n_jobs: int = -1

    clim: Tuple[float, float] = (-1.0, 1.0)
    cmap: str = "jet"


# =========================================================
# Utilities
# =========================================================
def build_boundaries(celltypes: np.ndarray) -> pd.DataFrame:
    celltypes = np.asarray(celltypes, dtype=str)
    change = np.r_[True, celltypes[1:] != celltypes[:-1]]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:] - 1, celltypes.size - 1]
    return pd.DataFrame({"CellType": celltypes[starts], "Start": starts, "End": ends})


def _clean_blocks(blocks: List[np.ndarray], n: int) -> List[np.ndarray]:
    clean = []
    for v in blocks:
        if v is None:
            continue
        v = np.unique(np.asarray(v, dtype=int))
        v = v[(v >= 0) & (v < n)]
        if v.size >= 1:
            clean.append(v)
    return clean


def _sparsify(mat: np.ndarray, thr: float) -> np.ndarray:
    if thr <= 0:
        return mat
    m = mat.copy()
    m[np.abs(m) < thr] = 0.0
    return m


def _concat_preserve_order(blocks: List[np.ndarray], n: int) -> np.ndarray:
    seen = np.zeros(n, dtype=bool)
    out = []
    for b in blocks:
        for x in b:
            if not seen[x]:
                seen[x] = True
                out.append(x)
    return np.asarray(out, dtype=int)


# =========================================================
# Per-cell-type processing
# =========================================================
def _process_one_celltype(corr, row, opts):
    ct = str(row.CellType)
    s, e = int(row.Start), int(row.End)
    n = e - s + 1
    global_map = np.arange(s, e + 1)

    if n < opts.min_cells:
        return ct, [], global_map.copy(), []

    sub = corr[s:e + 1, s:e + 1].copy()
    np.fill_diagonal(sub, 0)

    sub1 = _sparsify(sub, opts.corr_threshold)
    blocks1, _ = greedy_peeling_X_all(sub1, opts.lambda_stage1)
    blocks1 = _clean_blocks(blocks1, n)
    blocks1 = [b for b in blocks1 if b.size >= opts.min_block_size]

    if len(blocks1) == 0:
        blocks1 = [np.arange(n)]

    final_blocks = list(blocks1)

    if len(blocks1) == 1:
        remain = np.setdiff1d(np.arange(n), blocks1[0])
        if remain.size >= opts.min_cells:
            sub2 = sub[np.ix_(remain, remain)]
            if np.mean(np.abs(sub2)) >= opts.stage2_skip_mean_abs:
                sub2 = _sparsify(sub2, opts.corr_threshold)
                blocks2, _ = greedy_peeling_X_all(sub2, opts.lambda_stage2)
                blocks2 = _clean_blocks(blocks2, remain.size)
                blocks2 = [remain[b] for b in blocks2 if b.size >= opts.min_block_size]
                final_blocks += blocks2

    global_blocks = [global_map[b] for b in final_blocks]

    rows = []
    for i, g in enumerate(global_blocks, 1):
        rows.append({
            "CellType": ct,
            "SubBlockID": i,
            "NumBlocksInCT": len(global_blocks),
            "BlockSize": g.size,
            "CT_Start": s,
            "CT_End": e,
            "Block_Start": int(g.min()),
            "Block_End": int(g.max())
        })

    clist = _concat_preserve_order(final_blocks, n)
    other = np.setdiff1d(np.arange(n), clist)
    order = global_map[np.r_[clist, other]]

    return ct, global_blocks, order, rows


# =========================================================
# Side-by-side plotting (FIXED)
# =========================================================
def _plot_side_by_side(corr, boundaries, blocks, order, opts):
    corr_sorted = corr[np.ix_(order, order)]
    pos = {g: i for i, g in enumerate(order)}

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 6), dpi=150,
        gridspec_kw={"wspace": 0.12}
    )

    # ---------- LEFT: original ----------
    im = axes[0].imshow(
        corr, cmap=opts.cmap,
        vmin=opts.clim[0], vmax=opts.clim[1]
    )
    axes[0].set_title("Original correlation (cell-type boxes)")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    colors = plt.cm.tab10(np.linspace(0, 1, len(boundaries)))
    for i, r in boundaries.iterrows():
        s, e = r.Start, r.End
        axes[0].add_patch(
            patches.Rectangle(
                (s - .5, s - .5),
                e - s + 1, e - s + 1,
                edgecolor=colors[i],
                facecolor="none", lw=2
            )
        )

    # ---------- RIGHT: sorted ----------
    axes[1].imshow(
        corr_sorted, cmap=opts.cmap,
        vmin=opts.clim[0], vmax=opts.clim[1]
    )
    axes[1].set_title("Sorted correlation (cell-type boxes + detected blocks)")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    start = 0
    for i, r in boundaries.iterrows():
        n = r.End - r.Start + 1
        axes[1].add_patch(
            patches.Rectangle(
                (start - .5, start - .5),
                n, n,
                edgecolor=colors[i],
                facecolor="none", lw=2
            )
        )
        start += n

    for g in blocks:
        idx = np.array([pos[x] for x in g])
        s, e = idx.min(), idx.max()
        axes[1].add_patch(
            patches.Rectangle(
                (s - .5, s - .5),
                e - s + 1, e - s + 1,
                edgecolor="red",
                facecolor="none", lw=1
            )
        )

    # ---------- shared colorbar ----------
    cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


# =========================================================
# Main entry
# =========================================================
def cxc_refinement_py(corr, celltypes, opts=None):
    if opts is None:
        opts = CXCOptions()

    corr = np.asarray(corr, float)
    corr[np.isnan(corr)] = 0
    np.fill_diagonal(corr, 0)

    boundaries = build_boundaries(celltypes)

    global_blocks = []
    rows = []
    order_full = []

    for _, r in boundaries.iterrows():
        _, blocks, order, info = _process_one_celltype(corr, r, opts)
        global_blocks.extend(blocks)
        order_full.append(order)
        rows.extend(info)

    order_full = np.concatenate(order_full)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["GlobalBlockID"] = np.arange(1, len(df) + 1)

    fig_png = _plot_side_by_side(
        corr, boundaries, global_blocks, order_full, opts
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("detected_blocks_summary.csv", df.to_csv(index=False))
    buf.seek(0)

    return {
        "global_blockinfo_df": df,
        "fig_side_by_side_png": fig_png,
        "artifacts_zip_bytes": buf.getvalue()
    }