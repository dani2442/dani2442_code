"""A deterministic reference solution, so the Monte Carlo has something to miss.

Laplace's equation on the bean with Dirichlet data `g` is discretized on a
uniform Cartesian grid with the Shortley-Weller stencil: a node whose neighbour
falls outside the domain uses the *actual* distance to {phi = 0} along that grid
line (found by bisection) instead of the grid spacing, so the boundary is
resolved to second order rather than staircased.

Two things come out of one factorization:

  * `u`, the field itself -- the object the probabilistic formula represents;
  * `weights`, the discrete exit distribution seen from a single node.  Writing
    the discrete problem as `A u = -B g`, the value at a node is
    `u(x0) = e^T A^{-1} (-B g) = <w, g>` with `w = -B^T A^{-T} e`, so one
    transposed solve returns the harmonic measure of `x0` as a vector of
    nonnegative weights on the boundary points -- Feynman-Kac without any
    sampling.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


class Reference:
    def __init__(self, bean, n=500):
        self.bean = bean
        x0, x1, y0, y1 = bean.bbox
        self.h = (x1 - x0) / (n - 1)
        self.x = x0 + self.h * np.arange(n)
        self.y = np.arange(y0, y1 + self.h, self.h)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.P = np.stack([self.X, self.Y], axis=-1)
        self.inside = bean.phi(self.P) < 0.0
        self.ny, self.nx = self.inside.shape
        if (self.inside[0].any() or self.inside[-1].any()
                or self.inside[:, 0].any() or self.inside[:, -1].any()):
            raise ValueError("bbox does not enclose the domain with a margin")

        self.ids = -np.ones(self.inside.shape, dtype=int)
        self.ids[self.inside] = np.arange(self.inside.sum())
        self.n_unknown = int(self.inside.sum())
        self._assemble()

    # -- assembly --------------------------------------------------------
    def _assemble(self):
        """Build A and B with the non-uniform (Shortley-Weller) stencil."""
        h = self.h
        ii, jj = np.nonzero(self.inside)
        node = self.ids[ii, jj]

        # Distance to the next grid point in each direction: h if that point is
        # inside, otherwise the bisected distance to the boundary.
        dist = np.full((4, len(node)), h)
        out_mask = np.zeros((4, len(node)), dtype=bool)
        hit = {}
        for d, (dj, di) in enumerate(DIRS):
            out = ~self.inside[ii + di, jj + dj]
            if not out.any():
                continue
            a = self.P[ii[out], jj[out]]
            b = a + h * np.array([dj, di], dtype=float)
            xb = self.bean.crossing(a, b)
            dist[d, out] = np.hypot(*(xb - a).T)
            out_mask[d] = out
            hit[d] = xb

        rows, cols, vals = [], [], []
        brows, bcols, bvals, bxy = [], [], [], []
        ncol = 0
        for dp, dm in [(0, 1), (2, 3)]:
            hp, hm = dist[dp], dist[dm]
            cp = 2.0 / (hp * (hp + hm))
            cm = 2.0 / (hm * (hp + hm))
            rows.append(node); cols.append(node); vals.append(-(cp + cm))
            for d, c in ((dp, cp), (dm, cm)):
                dj, di = DIRS[d]
                out = out_mask[d]
                if out.any():
                    m = len(hit[d])
                    brows.append(node[out])
                    bcols.append(ncol + np.arange(m))
                    bvals.append(c[out])
                    bxy.append(hit[d])
                    ncol += m
                inn = ~out
                rows.append(node[inn])
                cols.append(self.ids[ii[inn] + di, jj[inn] + dj])
                vals.append(c[inn])

        n = self.n_unknown
        self.A = sp.coo_matrix((np.concatenate(vals),
                                (np.concatenate(rows), np.concatenate(cols))),
                               shape=(n, n)).tocsc()
        self.bnd_xy = np.vstack(bxy)
        self.B = sp.coo_matrix((np.concatenate(bvals),
                                (np.concatenate(brows), np.concatenate(bcols))),
                               shape=(n, ncol)).tocsc()
        self.lu = spla.splu(self.A)

    # -- solves ----------------------------------------------------------
    def solve(self):
        """The Dirichlet solution, as a masked field on the grid."""
        gb = self.bean.g(self.bnd_xy)
        u = self.lu.solve(-(self.B @ gb))
        field = np.full(self.inside.shape, np.nan)
        field[self.inside] = u
        self.u = u
        self.field = field
        return field

    def exit_time(self):
        """E_x[tau]: the Poisson problem (1/2) Delta v = -1, v = 0 on dOmega."""
        v = self.lu.solve(-2.0 * np.ones(self.n_unknown))
        field = np.full(self.inside.shape, np.nan)
        field[self.inside] = v
        self.tau = v
        return field

    def node(self, x0):
        """Index of the interior grid node nearest to `x0` (and that node)."""
        d = np.hypot(self.X - x0[0], self.Y - x0[1])
        d[~self.inside] = np.inf
        i, j = np.unravel_index(np.argmin(d), d.shape)
        return self.ids[i, j], self.P[i, j]

    def value(self, x0):
        k, xg = self.node(x0)
        if not hasattr(self, "u"):
            self.solve()
        return self.u[k], xg

    def weights(self, x0):
        """Discrete harmonic measure of `x0`: weights on `self.bnd_xy`."""
        k, xg = self.node(x0)
        e = np.zeros(self.n_unknown)
        e[k] = 1.0
        w = -(self.B.T @ self.lu.solve(e, trans="T"))
        return np.asarray(w).ravel(), xg
