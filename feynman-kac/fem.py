"""A P1 finite element solve on an unstructured mesh of the bean.

The rest of the post computes the deterministic side with finite differences on
a Cartesian grid (`reference.py`), which is the right tool for the harmonic
measure but has no mesh to look at.  This file exists for the picture: a genuine
triangulation of the curved domain and a genuine Galerkin solve on it, so the
figure showing "the FEM solution on its mesh" is showing exactly that.

Mesh: the boundary curve resampled at spacing `h`, plus a hexagonal lattice of
interior nodes kept clear of the curve, Delaunay-triangulated and then cut back
to the *polygon* those boundary nodes span -- which is also the domain the
Galerkin solve really sees.  Cutting against `phi` instead would be wrong at the
concave notch, where the chord between two neighbouring boundary nodes lies
outside `{phi < 0}`; the polygon test keeps exactly the cells that tile it, and
still discards the fans the convex hull throws across the notch.  A few sweeps
of Laplacian smoothing even out the row of cells next to the boundary.

Solve: linear Lagrange elements, exact for the piecewise-constant gradients,
with the Dirichlet data imposed by elimination:

    K_ff u_f = -K_fc g(x_c),      K_ij = sum_T |T| grad(phi_i).grad(phi_j)
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from matplotlib.path import Path
from scipy.spatial import Delaunay

from domain import Arclength


# -----------------------------------------------------------------------------
# mesh
# -----------------------------------------------------------------------------
def _resample(poly, h):
    """Points along a closed polyline at (almost exactly) spacing `h`."""
    al = Arclength(poly)
    m = max(int(round(al.total / h)), 8)
    s = np.linspace(0.0, al.total, m, endpoint=False)
    j = np.clip(np.searchsorted(al.cum, s, side="right") - 1, 0, len(al.seg) - 1)
    t = (s - al.cum[j]) / np.hypot(*al.seg[j].T)
    return poly[j] + t[:, None] * al.seg[j]


def _interior(bean, h, margin=0.62):
    """Hexagonal lattice of nodes at least `margin * h` inside the domain."""
    x0, x1, y0, y1 = bean.bbox
    dy = h * np.sqrt(3) / 2
    rows = []
    for r, y in enumerate(np.arange(y0, y1 + dy, dy)):
        x = np.arange(x0 + (0.5 * h if r % 2 else 0.0), x1 + h, h)
        rows.append(np.stack([x, np.full_like(x, y)], axis=1))
    p = np.vstack(rows)
    return p[bean.phi(p) < -margin * h]


def _cut(bnd, pts, tri):
    """Keep the triangles whose centroid lies inside the boundary polygon."""
    poly = Path(np.vstack([bnd, bnd[:1]]), closed=True)
    return tri[poly.contains_points(pts[tri].mean(axis=1))]


def _smooth(pts, tri, n_bnd, rounds=12, omega=0.55):
    """Laplacian smoothing of the free nodes; boundary nodes are pinned."""
    n = len(pts)
    e = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    e = np.vstack([e, e[:, ::-1]])
    W = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])),
                      shape=(n, n)).tocsr()
    W.data[:] = 1.0                                # collapse duplicate edges
    deg = np.asarray(W.sum(axis=1)).ravel()
    free = np.zeros(n, dtype=bool)
    free[n_bnd:] = deg[n_bnd:] > 0
    p = pts.copy()
    for _ in range(rounds):
        bary = (W @ p) / np.maximum(deg, 1)[:, None]
        p[free] += omega * (bary[free] - p[free])
    return p


def mesh(bean, h=0.05):
    """Nodes (boundary ones first) and triangles of a mesh of the bean."""
    bnd = _resample(bean.boundary(1400), h)
    pts = np.vstack([bnd, _interior(bean, h)])
    tri = _cut(bnd, pts, Delaunay(pts).simplices)
    pts = _smooth(pts, tri, len(bnd))
    tri = _cut(bnd, pts, Delaunay(pts).simplices)

    used = np.zeros(len(pts), dtype=bool)
    used[tri.ravel()] = True
    used[:len(bnd)] = True                         # the curve stays whole
    keep = np.nonzero(used)[0]
    relabel = -np.ones(len(pts), dtype=int)
    relabel[keep] = np.arange(len(keep))
    return pts[keep], relabel[tri], len(bnd)


# -----------------------------------------------------------------------------
# solve
# -----------------------------------------------------------------------------
def stiffness(pts, tri):
    """Global P1 stiffness matrix of `-Delta`."""
    v = pts[tri]
    x, y = v[..., 0], v[..., 1]
    b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], 1)
    c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], 1)
    det = b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0]    # = 2 * signed area
    area = 0.5 * np.abs(det)
    k = (b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :])
    k /= (2.0 * np.abs(det))[:, None, None]
    rows = np.repeat(tri, 3, axis=1).ravel()
    cols = np.tile(tri, 3).ravel()
    n = len(pts)
    K = sp.coo_matrix((k.ravel(), (rows, cols)), shape=(n, n)).tocsr()
    return K, area


def solve(bean, h=0.05):
    """Solve `Delta u = 0`, `u = g` on the boundary, with P1 elements."""
    pts, tri, n_bnd = mesh(bean, h)
    K, area = stiffness(pts, tri)
    u = np.empty(len(pts))
    u[:n_bnd] = bean.g(pts[:n_bnd])
    f = slice(n_bnd, len(pts))
    u[f] = spla.spsolve(K[f, f].tocsc(), -(K[f, :n_bnd] @ u[:n_bnd]))
    return pts, tri, u, n_bnd, area


def value_at(pts, tri, u, x):
    """The finite element solution at one point, by its barycentric coordinates."""
    d = Delaunay(pts)
    t = d.find_simplex(np.atleast_2d(x))[0]
    v = pts[d.simplices[t]]
    M = np.vstack([v.T, np.ones(3)])
    lam = np.linalg.solve(M, np.r_[np.asarray(x, dtype=float), 1.0])
    return float(lam @ u[d.simplices[t]])
