import io
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

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
    # thresholds for dense pattern detection
    lambda_stage1: float = 1.93
    lambda_stage2: float = 1.92

    # minimum cells per cell type to run detection
    min_cells: int = 30

    # -------------------------
    # SPEED / ROBUSTNESS
    # -------------------------
    # sparsify weak correlations before greedy peeling
    corr_threshold: float = 0.05

    # stage-2: skip if remaining region too flat
    stage2_skip_mean_abs: float = 0.02

    # parallel per cell type
    parallel: bool = False
    n_jobs: int = -1

    # -------------------------
    # Plotting
    # -------------------------
    clim: Tuple[float, float] = (-1.0, 1.0)
    cmap: str = "jet"  # MATLAB jet


# =========================================================
# Utilities
# =========================================================
def build_boundaries(celltypes: np.ndarray) -> pd.DataFrame:
    """
    Build contiguous cell-type boundaries (0-based, inclusive).
    Assumes celltypes are already grouped contiguously by cell type.
    """
    celltypes = np.asarray(celltypes, dtype=str)
    change = np.r_[True, celltypes[1:] != celltypes[:-1]]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:] - 1, celltypes.size - 1]
    return pd.DataFrame({"CellType": celltypes[starts], "Start": starts, "End": ends})


def _clean_blocks(blocks: List[np.ndarray], n: int) -> List[np.ndarray]:
    """
    Clean greedy-peeling blocks: unique, in-range, non-empty.
    """
    clean = []
    for v in blocks:
        if v is None:
            continue
        v = np.asarray(v, dtype=int).ravel()
        v = np.unique(v)
        v = v[(v >= 0) & (v < n)]
        if v.size > 0:
            clean.append(v)
    return clean


def _sparsify(mat: np.ndarray, thr: float) -> np.ndarray:
    """
    Zero out weak correlations to speed greedy peeling.
    """
    if thr <= 0:
        return mat
    m = mat.copy()
    m[np.abs(m) < thr] = 0.0
    return m


def _concat_preserve_order(blocks: List[np.ndarray], n: int) -> np.ndarray:
    """
    Concatenate blocks preserving order and first occurrence.
    This is critical to keep block contiguity in sorted plots.
    """
    seen = np.zeros(n, dtype=bool)
    out = []
    for b in blocks:
        for x in np.asarray(b, dtype=int).ravel():
            if 0 <= x < n and not seen[x]:
                seen[x] = True
                out.append(x)
    return np.asarray(out, dtype=int)


# =========================================================
# Per-celltype worker
# =========================================================
def _process_one_celltype(
    corr: np.ndarray,
    row: pd.Series,
    opts: CXCOptions
) -> Tuple[str, int, int, List[np.ndarray], np.ndarray, np.ndarray, List[Dict]]:
    """
    Returns:
      ct, s, e,
      global_blocks_for_ct (list of global index arrays),
      local_order (0-based within CT),
      global_order (global indices),
      block_rows (summary rows)
    """
    ct = str(row.CellType)
    s, e = int(row.Start), int(row.End)
    n = e - s + 1
    global_map = np.arange(s, e + 1, dtype=int)

    # If CT is too small, skip detection
    if n < opts.min_cells:
        local_order = np.arange(n, dtype=int)
        global_order = global_map[local_order]
        return ct, s, e, [], local_order, global_order, []

    # Extract CT submatrix once
    sub = corr[s:e + 1, s:e + 1].copy()
    np.fill_diagonal(sub, 0.0)

    # Speed: sparsify before stage-1
    sub_work = _sparsify(sub, opts.corr_threshold)

    # =====================================================
    # Stage 1: detect on full CT
    # =====================================================
    blocks1, _ = greedy_peeling_X_all(sub_work, opts.lambda_stage1)
    clean1 = _clean_blocks(blocks1, n)

    # remove trivial singletons
    clean1 = [b for b in clean1 if b.size > 1]

    # fallback if nothing meaningful found
    if len(clean1) == 0:
        clean1 = [np.arange(n, dtype=int)]

    # =====================================================
    # Decision + Stage 2 (your rule)
    # =====================================================
    if len(clean1) > 1:
        final_blocks = clean1
    else:
        # exactly ONE stage-1 block
        stage1_block = clean1[0]
        remaining = np.setdiff1d(np.arange(n, dtype=int), stage1_block)

        if remaining.size < opts.min_cells:
            final_blocks = clean1
        else:
            # stage-2 on remaining rows/cols ONLY (exclude stage-1 block)
            sub2 = sub[np.ix_(remaining, remaining)]

            # optional quick skip (flat remainder)
            if np.mean(np.abs(sub2)) < opts.stage2_skip_mean_abs:
                final_blocks = clean1
            else:
                sub2_work = _sparsify(sub2, opts.corr_threshold)
                blocks2, _ = greedy_peeling_X_all(sub2_work, opts.lambda_stage2)
                clean2 = _clean_blocks(blocks2, remaining.size)

                # map stage-2 indices back to CT-local indices
                clean2 = [remaining[b] for b in clean2]

                # remove singletons
                clean2 = [b for b in clean2 if b.size > 1]

                # Your condition:
                # "if stage 2 detected one block with size = 1 -> keep stage 1"
                # In practice after filtering size>1, that means clean2 becomes empty.
                # Also handle cases where stage-2 finds nothing meaningful.
                if len(clean2) == 0:
                    final_blocks = clean1
                else:
                    final_blocks = clean1 + clean2

    # =====================================================
    # Save blocks (global indices) + summary rows
    # =====================================================
    global_blocks_for_ct: List[np.ndarray] = []
    block_rows: List[Dict] = []

    for sub_id, b in enumerate(final_blocks, start=1):
        gidx = global_map[b]
        global_blocks_for_ct.append(gidx)
        block_rows.append({
            "CellType": ct,
            "SubBlockID": sub_id,
            "NumBlocksInCT": len(final_blocks),
            "BlockSize": int(gidx.size),
            "CT_Start": s,
            "CT_End": e
        })

    # =====================================================
    # Ordering (block-first, preserve contiguity)
    # =====================================================
    clist = _concat_preserve_order(final_blocks, n=n)
    other = np.setdiff1d(np.arange(n, dtype=int), clist)
    local_order = np.r_[clist, other]
    global_order = global_map[local_order]

    return ct, s, e, global_blocks_for_ct, local_order, global_order, block_rows


# =========================================================
# Main refinement
# =========================================================
def cxc_refinement_py(
    corr: np.ndarray,
    celltypes: np.ndarray,
    opts: Optional[CXCOptions] = None
) -> Dict:
    if opts is None:
        opts = CXCOptions()

    corr = np.asarray(corr, float)
    corr[np.isnan(corr)] = 0.0
    np.fill_diagonal(corr, 0.0)

    celltypes = np.asarray(celltypes, dtype=str)
    boundaries_df = build_boundaries(celltypes)

    # per-CT processing
    rows = [r for _, r in boundaries_df.iterrows()]
    if opts.parallel:
        try:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=opts.n_jobs, backend="loky")(
                delayed(_process_one_celltype)(corr, r, opts) for r in rows
            )
        except Exception:
            results = [_process_one_celltype(corr, r, opts) for r in rows]
    else:
        results = [_process_one_celltype(corr, r, opts) for r in rows]

    # collect outputs
    global_blocks: List[np.ndarray] = []
    block_rows: List[Dict] = []
    order_files_local: Dict[str, np.ndarray] = {}
    order_files_global: Dict[str, np.ndarray] = {}

    global_block_id = 0
    for (ct, s, e, blocks_ct, local_order, global_order, rows_ct) in results:
        order_files_local[ct] = local_order
        order_files_global[ct] = global_order

        # assign global IDs in stable order per CT
        for sub_id, gidx in enumerate(blocks_ct, start=1):
            global_block_id += 1
            global_blocks.append(gidx)

        for rr in rows_ct:
            # assign GlobalBlockID in same order as SubBlockID per CT
            # (stable even when some CTs have 0 blocks stored)
            rr2 = dict(rr)
            # approximate assignment: global IDs are not strictly needed for your plots,
            # but we keep them for consistency in outputs
            rr2["GlobalBlockID"] = None
            block_rows.append(rr2)

    blockinfo_df = pd.DataFrame(block_rows)
    if blockinfo_df.shape[0] > 0:
        ct_order = boundaries_df["CellType"].tolist()
        blockinfo_df["CellType"] = pd.Categorical(
            blockinfo_df["CellType"], categories=ct_order, ordered=True
        )
        blockinfo_df = (
            blockinfo_df
            .sort_values(["CellType", "SubBlockID"], kind="stable")
            .reset_index(drop=True)
        )
        blockinfo_df["GlobalBlockID"] = np.arange(1, len(blockinfo_df) + 1, dtype=int)
    else:
        blockinfo_df = pd.DataFrame(columns=[
            "GlobalBlockID", "CellType", "SubBlockID", "NumBlocksInCT",
            "BlockSize", "CT_Start", "CT_End"
        ])

    # ZIP outputs
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("detected_blocks_summary.csv", blockinfo_df.to_csv(index=False))
        for ct, order in order_files_local.items():
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in ct)
            z.writestr(
                f"order_{safe}.csv",
                pd.DataFrame({"local_order_0based": order}).to_csv(index=False)
            )
    zip_buf.seek(0)

    # Figures
    fig_original_png = _plot_original(corr, boundaries_df, opts)
    fig_sorted_png = _plot_sorted(corr, boundaries_df, global_blocks, order_files_global, opts)

    return {
        "global_blockinfo_df": blockinfo_df,
        "fig_original_png": fig_original_png,
        "fig_sorted_png": fig_sorted_png,
        "artifacts_zip_bytes": zip_buf.getvalue()
    }


# =========================================================
# Plot helpers
# =========================================================
def _plot_original(corr, boundaries_df, opts):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    im = ax.imshow(corr, cmap=opts.cmap, vmin=opts.clim[0], vmax=opts.clim[1])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    colors = plt.cm.tab10(np.linspace(0, 1, len(boundaries_df)))
    for i, row in boundaries_df.iterrows():
        s, e = int(row.Start), int(row.End)
        ax.add_patch(patches.Rectangle(
            (s - 0.5, s - 0.5),
            e - s + 1, e - s + 1,
            edgecolor=colors[i], facecolor="none", linewidth=2
        ))

    ax.set_title("Original correlation (cell-type boxes)")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _plot_sorted(corr, boundaries_df, global_blocks, order_files_global, opts):
    # IMPORTANT: build GLOBAL order (not CT-local indices)
    order_full = np.concatenate([order_files_global[ct] for ct in boundaries_df.CellType])

    corr_sorted = corr[np.ix_(order_full, order_full)]

    # map global index -> position in sorted matrix
    pos_map = np.empty(corr.shape[0], dtype=int)
    pos_map[order_full] = np.arange(order_full.size)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    im = ax.imshow(corr_sorted, cmap=opts.cmap, vmin=opts.clim[0], vmax=opts.clim[1])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    colors = plt.cm.tab10(np.linspace(0, 1, len(boundaries_df)))

    # cell-type boxes
    pos = 0
    for i, row in boundaries_df.iterrows():
        n = int(row.End - row.Start + 1)
        ax.add_patch(patches.Rectangle(
            (pos - 0.5, pos - 0.5), n, n,
            edgecolor=colors[i], facecolor="none", linewidth=2
        ))
        pos += n

    # detected blocks (red rectangles)
    for gidx in global_blocks:
        ridx = pos_map[gidx]
        s = int(ridx.min())
        e = int(ridx.max())
        w = e - s + 1
        ax.add_patch(patches.Rectangle(
            (s - 0.5, s - 0.5), w, w,
            edgecolor="red", facecolor="none", linewidth=1
        ))

    ax.set_title("Sorted correlation (cell-type boxes + detected blocks)")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()