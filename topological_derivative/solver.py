"""
The optimization loop.

One iteration solves the elasticity problem on the current design, prices hole
nucleation everywhere with the topological derivative, regularizes that field
with a cone filter, and keeps the highest-scoring cells up to the volume
allowed by a shrinking schedule.  Nothing here knows about a particular load
case: everything it needs comes from the `Problem` it is handed.
"""

import numpy as np
import scipy.sparse as sp

import fem
from problem import E_SOLID, E_MIN, NU, MODEL


# -----------------------------------------------------------------------------
# Cone filter on the cell grid
# -----------------------------------------------------------------------------
def cone_filter(nx, ny, hx, hy, rmin):
    """Row-normalized linear-hat (cone) averaging operator of radius rmin.

    The filter both fixes the minimum feature size and extends the solid-phase
    field into the void, which is what lets material be re-inserted.
    """
    idx = np.arange(nx * ny).reshape(nx, ny)
    rows, cols, vals = [], [], []
    for di in range(-int(rmin / hx), int(rmin / hx) + 1):
        for dj in range(-int(rmin / hy), int(rmin / hy) + 1):
            w = rmin - np.hypot(di * hx, dj * hy)
            if w <= 0.0:
                continue
            i0, i1 = max(0, -di), min(nx, nx - di)
            j0, j1 = max(0, -dj), min(ny, ny - dj)
            if i0 >= i1 or j0 >= j1:
                continue
            src = idx[i0:i1, j0:j1].ravel()
            dst = idx[i0 + di:i1 + di, j0 + dj:j1 + dj].ravel()
            rows.append(dst)
            cols.append(src)
            vals.append(np.full(src.size, w))
    H = sp.coo_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(nx * ny, nx * ny)).tocsr()
    return sp.diags(1.0 / np.asarray(H.sum(axis=1)).ravel()) @ H


# -----------------------------------------------------------------------------
# One iteration
# -----------------------------------------------------------------------------
def solve_state(p, chi):
    """Solve on the design `chi` and return (u, J, per-triangle D_T J)."""
    E_cell = np.where(chi > 0.5, E_SOLID, E_MIN)
    E_cell[p.void] = E_MIN
    u, J = p.pb.solve(E_cell[p.cell], p.f, p.fixed)
    # The formula is exact at points of the solid phase, where the recovered
    # stress is the physical one; evaluate it with the solid moduli everywhere
    # and mask the void afterwards.
    sigma = p.pb.stress(u, E_SOLID)
    return u, J, fem.topological_derivative(sigma, E_SOLID, NU, MODEL)


def _score(p, H, chi, g_tri, g_prev):
    """Cell-wise topological gradient, filtered and temporally averaged.

    Averaging with the previous score is the recursive form of the BESO
    temporal smoothing: what is stored is already an average, so the smoothing
    accumulates over the whole run.
    """
    g_cell = np.zeros(p.n_cells)
    np.add.at(g_cell, p.cell, 0.5 * g_tri)  # two equal-area triangles per cell
    g = H @ (g_cell * chi)                  # rigorous only on the solid phase
    return g if g_prev is None else 0.5 * (g + g_prev)


def _update(p, g, n_keep):
    """Keep the `n_keep` highest-scoring cells, protected ones first.

    This is the Lagrangian form of the volume constraint: for L = J + l |Omega|
    the change under hole nucleation is |B|(D_T J - l), so cells with
    D_T J < l should be voided, and the multiplier l is exactly the volume
    quantile of the filtered score that this ranking cuts at.
    """
    chi = np.zeros(p.n_cells)
    chi[p.keep & ~p.void] = 1.0
    n_forced = int(chi.sum())
    free = np.where(~p.void & ~p.keep)[0]
    order = free[np.argsort(-g[free])]
    chi[order[:max(0, n_keep - n_forced)]] = 1.0
    return chi


# -----------------------------------------------------------------------------
# The loop
# -----------------------------------------------------------------------------
def optimize(p, verbose=True, record=False):
    """Return the final design and the history of solved states.

    With record=True the history also retains each design and the nodal
    displacement magnitude |u| of the state solved on it, so a GIF frame always
    pairs a layout with its own deflection.
    """
    n_iter = p.n_iter
    hx, hy = p.h
    H = cone_filter(p.nx, p.ny, hx, hy, p.rmin_cells * max(hx, hy))

    design_area = float((~p.void).sum())
    chi = (~p.void).astype(float)
    g_prev = None
    hist = {"J": [], "V": []}
    if record:
        hist.update(chi=[], U=[])

    for k in range(n_iter):
        u, J, g_tri = solve_state(p, chi)
        g = _score(p, H, chi, g_tri, g_prev)
        g_prev = g

        V = chi.sum() / design_area
        hist["J"].append(J)
        hist["V"].append(V)
        if record:
            hist["chi"].append(chi.copy())
            hist["U"].append(fem.displacement_magnitude(u))
        if verbose and (k % 10 == 0 or k == n_iter - 1):
            print(f"   it {k:3d}   V = {V:5.3f}   J = {J:.6g}")

        # Shrink the volume, then hold it while the topology settles.
        V_next = max(p.vol_frac, V * (1.0 - p.evol_rate))
        chi = _update(p, g, int(round(V_next * design_area)))

    # A fresh state solve on the final design, so the last recorded frame is a
    # solved state like every other one.
    u, J, _ = solve_state(p, chi)
    hist["J"].append(J)
    hist["V"].append(chi.sum() / design_area)
    if record:
        hist["chi"].append(chi.copy())
        hist["U"].append(fem.displacement_magnitude(u))
    if verbose:
        print(f"   final  V = {hist['V'][-1]:5.3f}   J = {J:.6g}")
    return chi, hist
