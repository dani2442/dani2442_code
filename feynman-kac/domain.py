"""The bean: a smooth planar domain given as the zero set of a level function.

The shape is built from three disks with smooth Boolean operators -- two lobes
smoothly unioned, one disk smoothly subtracted to carve the concave notch -- so
that `phi` is C^infinity, negative inside, positive outside, and has a
non-vanishing gradient on {phi = 0}.  Everything the rest of the code needs from
the geometry goes through this file: the inside test, the boundary crossing of a
segment, the boundary datum `g`, and an arclength parametrization of the curve
used for binning exit locations.
"""

import numpy as np


def _smin(a, b, k):
    """Smooth minimum (smooth union of two signed level functions)."""
    return -np.logaddexp(-k * a, -k * b) / k


def _smax(a, b, k):
    """Smooth maximum (smooth intersection)."""
    return np.logaddexp(k * a, k * b) / k


class Bean:
    """Bean-shaped domain Omega = {phi < 0}, with Dirichlet data g on dOmega."""

    def __init__(self, k_union=7.5, k_cut=18.0):
        self.k_union = k_union
        self.k_cut = k_cut
        self.lobes = [((-0.38, -0.05), 0.60), ((0.30, 0.02), 0.55),
                      ((0.62, 0.12), 0.42)]
        self.notch = ((0.10, 0.90), 0.66)
        self.bbox = (-1.10, 1.14, -0.72, 0.64)

    # -- level function --------------------------------------------------
    def phi(self, p):
        """Level function at points `p` of shape (..., 2); < 0 inside."""
        p = np.asarray(p, dtype=float)
        u = None
        for c, r in self.lobes:
            d = np.hypot(p[..., 0] - c[0], p[..., 1] - c[1]) - r
            u = d if u is None else _smin(u, d, self.k_union)
        (cn, rn) = self.notch
        dn = np.hypot(p[..., 0] - cn[0], p[..., 1] - cn[1]) - rn
        return _smax(u, -dn, self.k_cut)

    def inside(self, p):
        return self.phi(p) < 0.0

    def grad_phi(self, p, h=1e-5):
        """Outward normal direction (unnormalized), by central differences."""
        p = np.asarray(p, dtype=float)
        ex = np.zeros_like(p); ex[..., 0] = h
        ey = np.zeros_like(p); ey[..., 1] = h
        gx = (self.phi(p + ex) - self.phi(p - ex)) / (2 * h)
        gy = (self.phi(p + ey) - self.phi(p - ey)) / (2 * h)
        return np.stack([gx, gy], axis=-1)

    # -- geometry queries ------------------------------------------------
    def crossing(self, a, b, iters=30):
        """Point on {phi = 0} between `a` (inside) and `b` (outside), bisection.

        Vectorized over a leading batch axis.  Only the first crossing along the
        segment is found, which is what an exiting path needs.
        """
        a = np.array(a, dtype=float, copy=True)
        b = np.array(b, dtype=float, copy=True)
        for _ in range(iters):
            m = 0.5 * (a + b)
            hit = self.phi(m) < 0.0
            a = np.where(hit[..., None], m, a)
            b = np.where(hit[..., None], a * 0 + b, m)
        return 0.5 * (a + b)

    def g(self, p):
        """Dirichlet data on the boundary: a signed temperature profile."""
        p = np.asarray(p, dtype=float)
        return np.tanh(3.0 * p[..., 0] + 1.2 * p[..., 1])

    # -- meshes and the boundary curve -----------------------------------
    def grid(self, n=400):
        x0, x1, y0, y1 = self.bbox
        x = np.linspace(x0, x1, n)
        y = np.linspace(y0, y1, int(n * (y1 - y0) / (x1 - x0)))
        X, Y = np.meshgrid(x, y)
        return X, Y, self.phi(np.stack([X, Y], axis=-1))

    def boundary(self, n=900):
        """Closed polyline approximation of dOmega, counterclockwise."""
        from matplotlib import pyplot as plt
        X, Y, P = self.grid(n)
        fig = plt.figure()
        cs = fig.gca().contour(X, Y, P, levels=[0.0])
        segs = cs.allsegs[0]
        plt.close(fig)
        poly = max(segs, key=len)
        if np.hypot(*(poly[0] - poly[-1])) > 1e-9:
            poly = np.vstack([poly, poly[:1]])
        area = 0.5 * np.sum(poly[:-1, 0] * poly[1:, 1] - poly[1:, 0] * poly[:-1, 1])
        return poly if area > 0 else poly[::-1]


class Arclength:
    """Arclength coordinate on a closed polyline, plus nearest-point projection."""

    def __init__(self, poly):
        self.poly = poly
        seg = np.diff(poly, axis=0)
        self.seg = seg
        self.len2 = np.einsum("ij,ij->i", seg, seg)
        self.cum = np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
        self.total = self.cum[-1]

    def project(self, pts):
        """Arclength coordinate of the boundary point nearest to each of `pts`."""
        pts = np.atleast_2d(pts)
        d = pts[:, None, :] - self.poly[None, :-1, :]          # (n, m, 2)
        t = np.clip(np.einsum("nmj,mj->nm", d, self.seg) / self.len2, 0.0, 1.0)
        foot = self.poly[None, :-1, :] + t[..., None] * self.seg[None, :, :]
        j = np.argmin(np.einsum("nmj,nmj->nm", pts[:, None, :] - foot,
                                pts[:, None, :] - foot), axis=1)
        i = np.arange(len(pts))
        return self.cum[j] + t[i, j] * np.hypot(*self.seg[j].T)
