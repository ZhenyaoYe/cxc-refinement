import numpy as np

def greedy_peeling_X_one(W: np.ndarray, lam: float):
    """
    Weighted greedy peeling for a single densest subset.

    Returns:
      best_density: float
      best_nodes: 1D np.ndarray of indices (relative to input W) of best subset
    """
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be square")

    # Symmetry check (MATLAB used issymmetric)
    if not np.allclose(W, W.T, atol=1e-8, rtol=1e-8):
        raise ValueError("Matrix input must be symmetric")

    n = W.shape[0]
    nodes = np.arange(n, dtype=int)
    curW = W.copy()

    def density(M):
        m = M.shape[0]
        if m <= 0:
            return -np.inf
        return M.sum() / (m ** lam)

    best_density = density(curW)
    best_nodes = nodes.copy()

    # Peel until empty (or 1 node left)
    while curW.shape[0] > 1:
        deg = curW.sum(axis=1)
        rm = int(np.argmin(deg))

        # remove node rm
        nodes = np.delete(nodes, rm)
        curW = np.delete(np.delete(curW, rm, axis=0), rm, axis=1)

        d = density(curW)
        if d > best_density:
            best_density = d
            best_nodes = nodes.copy()

    return best_density, best_nodes


def greedy_peeling_X_all(Wp: np.ndarray, lam: float, max_rounds: int = 10):
    """
    Port of MATLAB greedy_peeling_X_all.m

    It repeatedly extracts densest subsets (by greedy_peeling_X_one) and peels them off,
    keeping only subsets whose density exceeds the ORIGINAL density computed on full Wp.

    Returns:
      blocks: list[np.ndarray]   each array are indices (0-based) into original Wp
      densities: list[float]
    """
    Wp = np.asarray(Wp, dtype=float)
    if Wp.ndim != 2 or Wp.shape[0] != Wp.shape[1]:
        raise ValueError("Wp must be square")
    if not np.allclose(Wp, Wp.T, atol=1e-8, rtol=1e-8):
        raise ValueError("Matrix input must be symmetric")

    m = Wp.shape[0]
    org_density = Wp.sum() / (m ** lam)

    X = Wp.copy()
    remain = np.arange(m, dtype=int)

    blocks = []
    densities = []

    i = 0
    while i < max_rounds and remain.size > 1:
        W_density, select_local = greedy_peeling_X_one(Wp, lam)

        if W_density > org_density and select_local.size > 0:
            # map local indices back to original
            chosen_global = remain[select_local]
            blocks.append(chosen_global)
            densities.append(W_density)

            # peel off
            all_chosen = np.concatenate(blocks) if blocks else np.array([], dtype=int)
            remain = np.setdiff1d(np.arange(m, dtype=int), all_chosen, assume_unique=False)

            Wp = X[np.ix_(remain, remain)]
            i += 1
        else:
            break

    return blocks, densities