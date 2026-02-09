import streamlit as st
import pandas as pd
import numpy as np

from cxc_refinement_py import cxc_refinement_py, CXCOptions


# =========================================================
# Page config
# =========================================================
st.set_page_config(layout="wide")
st.title("Cell-type Dense Pattern Refinement")

st.warning("All cells belonging to the same cell type must appear together (as a single contiguous block) in the correlation matrix.")

# =========================================================
# Session state
# =========================================================
if "corr_df" not in st.session_state:
    st.session_state.corr_df = None
if "ct_map" not in st.session_state:
    st.session_state.ct_map = None
if "cxc_results" not in st.session_state:
    st.session_state.cxc_results = None


# =========================================================
# Simulation utilities
# =========================================================
def simulate_correlation_matrix(
    cell_types=("Astrocyte", "Microglia", "Oligodendrocyte"),
    sizes=(1000, 500, 3000),
    seed=123
):
    rng = np.random.default_rng(seed)
    total_size = sum(sizes)

    # =====================================================
    # Global weak background correlation (across cell types)
    # =====================================================
    # This mimics shared technical / biological effects
    global_bg_mean = 0.08
    global_bg_sd   = 0.02

    mat = rng.normal(
        loc=global_bg_mean,
        scale=global_bg_sd,
        size=(total_size, total_size)
    )
    mat = (mat + mat.T) / 2
    np.fill_diagonal(mat, 1.0)

    idx = np.cumsum((0,) + sizes)

    # =====================================================
    # Cell-type internal block structure
    # =====================================================
    hierarchy = {
        0: dict(sub_sizes=[500, 500],   sub_means=[0.6, 0.4]),  # Astrocyte
        1: dict(sub_sizes=[500],        sub_means=[0.5]),       # Microglia
        2: dict(sub_sizes=[1500, 1500], sub_means=[0.6, 0.2]),  # Oligodendrocyte
    }

    # Sparsity controls (VERY important)
    within_block_density = 0.6   # fraction of non-zero edges inside blocks
    within_block_sd      = 0.05

    for i in range(len(sizes)):
        base = idx[i]
        sub_sizes = hierarchy[i]["sub_sizes"]
        sub_means = hierarchy[i]["sub_means"]
        sub_idx = np.cumsum([0] + sub_sizes)

        # -----------------------------
        # Within-subblock (SPARSE)
        # -----------------------------
        for k in range(len(sub_sizes)):
            s = base + sub_idx[k]
            e = base + sub_idx[k + 1]
            n = e - s

            # inflate mean so sparsity does NOT shrink correlations
            effective_mean = sub_means[k] / within_block_density

            # initialize empty block
            block = np.zeros((n, n))

            # apply sparsity mask
            mask = rng.random((n, n)) < within_block_density
            mask = np.triu(mask, k=1)
            # generate only non-zero edges
            vals = rng.normal(loc=effective_mean,
                              scale=within_block_sd,
                              size=mask.sum())
            block[mask] = vals

            # symmetrize + diagonal
            block = (block + block.T) / 2
            np.fill_diagonal(block, 1.0)

            mat[s:e, s:e] = block

        # -----------------------------
        # Between-subblocks (same CT)
        # -----------------------------
        for k in range(len(sub_sizes)):
            for l in range(k + 1, len(sub_sizes)):
                s1 = base + sub_idx[k]
                e1 = base + sub_idx[k + 1]
                s2 = base + sub_idx[l]
                e2 = base + sub_idx[l + 1]

                if i == 2:
                    # CT3: strong NEGATIVE coupling
                    mean_between = -0.8
                    sd_between   = 0.05
                else:
                    mean_between = 0.15
                    sd_between   = 0.03

                block = rng.normal(
                    loc=mean_between,
                    scale=sd_between,
                    size=(e1 - s1, e2 - s2)
                )

                mat[s1:e1, s2:e2] = block
                mat[s2:e2, s1:e1] = block.T

    # =====================================================
    # Final cleanup
    # =====================================================
    mat = np.clip(mat, -1, 1)

    # build labels
    cell_ids = []
    cell_type_vec = []
    for ct, n in zip(cell_types, sizes):
        cell_ids.extend([f"{ct}_{j+1}" for j in range(n)])
        cell_type_vec.extend([ct] * n)

    corr_df = pd.DataFrame(mat, index=cell_ids, columns=cell_ids)
    ct_map  = pd.DataFrame({"cell_id": cell_ids, "cell_type": cell_type_vec})

    return corr_df, ct_map


# =========================================================
# UI: data input
# =========================================================
mode = st.radio("Choose input mode:", ["Simulate data", "Upload custom data"])

if mode == "Simulate data":
    if st.button("Simulate correlation matrix"):
        st.session_state.corr_df, st.session_state.ct_map = simulate_correlation_matrix()
        st.session_state.cxc_results = None
        st.success("Simulated data generated.")
else:
    corr_file = st.file_uploader("Correlation matrix (.csv)", type=["csv"])
    ct_file = st.file_uploader("Cell-type mapping (.csv): must include columns cell_id, cell_type", type=["csv"])

    if corr_file and ct_file:
        st.session_state.corr_df = pd.read_csv(corr_file, index_col=0)
        st.session_state.ct_map = pd.read_csv(ct_file)
        st.session_state.cxc_results = None
        st.success("Files loaded successfully.")


# =========================================================
# CXC refinement (Python)
# =========================================================
st.divider()
st.header("CXC dense pattern refinement")

#col1, col2, col3 = st.columns(3)
col1, col2 = st.columns(2)

with col1:
    lambda1 = st.number_input("Lambda (primary )", value=1.93)
    lambda2 = st.number_input("Lambda (stage 2)", value=1.92)
    min_cells = st.number_input("Minimum cells required per cell type", value=30)

#with col2:
#    corr_thr = st.number_input("Speed: corr_threshold (sparsify)", value=0.05)
#    stage2_skip = st.number_input("Speed: stage2_skip_mean_abs", value=0.02)

with col2:
    parallel = st.checkbox("Parallel per cell type", value=False)
    n_jobs = st.number_input("n_jobs (parallel)", value=-1)
    clim_low = st.number_input("Heatmap clim min", value=-1.0)
    clim_high = st.number_input("Heatmap clim max", value=1.0)

run_btn = st.button("Run CXC refinement")

if run_btn:
    if st.session_state.corr_df is None or st.session_state.ct_map is None:
        st.error("Please simulate or upload data first.")
    else:
        st.info("Running refinement...")

        corr_df = st.session_state.corr_df
        ct_map = st.session_state.ct_map

        if "cell_id" in ct_map.columns:
            ct_map = ct_map.set_index("cell_id")
        if "cell_type" not in ct_map.columns:
            st.error("ct_map must contain a 'cell_type' column.")
            st.stop()

        missing = corr_df.index.difference(ct_map.index)
        if len(missing) > 0:
            st.error(f"ct_map missing {len(missing)} cell_id(s) from corr_df.")
            st.stop()

        celltypes = ct_map.loc[corr_df.index, "cell_type"].astype(str).values

        opts = CXCOptions(
            lambda_stage1=float(lambda1),
            lambda_stage2=float(lambda2),
            min_cells=int(min_cells),
            min_block_size=5,#int(min_block_size),
            corr_threshold=0.05, #float(corr_thr),
            stage2_skip_mean_abs=0.02, #float(stage2_skip),

            parallel=bool(parallel),
            n_jobs=int(n_jobs),

            clim=(float(clim_low), float(clim_high)),
            cmap="jet"  # MATLAB jet
        )

        res = cxc_refinement_py(corr_df.values, celltypes, opts)
        st.session_state.cxc_results = res
        st.success("CXC refinement completed!")


# =========================================================
# Show CXC outputs
# =========================================================
if st.session_state.cxc_results is not None:
    res = st.session_state.cxc_results

    st.subheader("Detected blocks summary")
    st.dataframe(res["global_blockinfo_df"], use_container_width=True)

    st.subheader("Figures")
    st.image(res["fig_original_png"], caption="Original correlation (cell-type boxes)")
    st.image(res["fig_sorted_png"], caption="Sorted correlation (cell-type boxes + detected blocks)")

    colA, colB = st.columns(2)
    with colA:
        st.download_button(
            label="Download original figure (PNG)",
            data=res["fig_original_png"],
            file_name="corr_original_celltype_boxes.png",
            mime="image/png"
        )
    with colB:
        st.download_button(
            label="Download sorted figure (PNG)",
            data=res["fig_sorted_png"],
            file_name="corr_sorted_blocks.png",
            mime="image/png"
        )

    st.subheader("Download all results")
    st.download_button(
        "Download outputs (ZIP: detected blocks summary, per-celltype order files)",
        data=res["artifacts_zip_bytes"],
        file_name="cxc_outputs.zip",
        mime="application/zip"
    )