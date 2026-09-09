"""
P1 (constant-strain triangle) plane elasticity, from scratch.

Companion code for the post *"The topological derivative and compliance
minimization"*.  Everything is built on `numpy` + `scipy.sparse`: structured
triangulations of a logical quad grid, CST assembly, Dirichlet elimination,
edge tractions, element stress recovery, and the topological-derivative field.

Conventions
-----------
Voigt order for the plane problem is  (xx, yy, xy)  with the *engineering*
shear strain  gamma_xy = 2 eps_xy  in the strain vector, so that
    sigma_voigt = D eps_voigt        and     sigma : eps = sigma_voigt . eps_voigt.
Displacement dofs are interleaved:  dof(2*i) = u_x(node i), dof(2*i+1) = u_y.
Unit out-of-plane thickness throughout.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# -----------------------------------------------------------------------------
# 1. Hooke's law for the two plane models
# -----------------------------------------------------------------------------
def hooke_2d(E, nu, model="plane_stress"):
    """Plane elasticity matrix D (3x3) in Voigt form."""
    if model == "plane_stress":
        return E / (1.0 - nu ** 2) * np.array([[1.0, nu, 0.0],
                                               [nu, 1.0, 0.0],
                                               [0.0, 0.0, 0.5 * (1.0 - nu)]])
    if model == "plane_strain":
        c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return c * np.array([[1.0 - nu, nu, 0.0],
                             [nu, 1.0 - nu, 0.0],
                             [0.0, 0.0, 0.5 - nu]])
    raise ValueError(f"unknown model {model!r}")


# -----------------------------------------------------------------------------
# 2. Structured triangulations
# -----------------------------------------------------------------------------
def structured_tri_mesh(X, Y):
    """Triangulate the structured quad grid with node coordinates X, Y.

    X, Y have shape (ni+1, nj+1).  Every quad is split into two triangles with
    the diagonal *alternating* in a union-jack pattern, so the mesh carries no
    global directional bias (a single fixed diagonal biases CST results, and
    biases the topology optimization along that diagonal).

    Returns
    -------
    nodes : (N, 2)      node coordinates
    tris  : (T, 3)      counter-clockwise vertex indices
    cell  : (T,)        index of the quad each triangle belongs to (row-major)
    shape : (ni, nj)    quad-grid shape
    """
    ni, nj = X.shape[0] - 1, X.shape[1] - 1
    nodes = np.column_stack([X.ravel(), Y.ravel()])
    nid = np.arange((ni + 1) * (nj + 1)).reshape(ni + 1, nj + 1)

    n00, n10 = nid[:-1, :-1], nid[1:, :-1]
    n11, n01 = nid[1:, 1:], nid[:-1, 1:]
    i = np.arange(ni)[:, None]
    j = np.arange(nj)[None, :]
    even = ((i + j) % 2 == 0)[..., None]

    t1 = np.where(even, np.stack([n00, n10, n11], -1), np.stack([n00, n10, n01], -1))
    t2 = np.where(even, np.stack([n00, n11, n01], -1), np.stack([n10, n11, n01], -1))
    tris = np.concatenate([t1.reshape(-1, 3), t2.reshape(-1, 3)], axis=0)
    cell = np.tile(np.arange(ni * nj), 2)

    # Force counter-clockwise orientation (positive signed area).
    x, y = nodes[tris, 0], nodes[tris, 1]
    two_a = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) \
          - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    flip = two_a < 0
    tris[flip] = tris[flip][:, [0, 2, 1]]
    return nodes, tris, cell, (ni, nj)


def rect_mesh(lx, ly, nx, ny):
    """Uniform triangulation of the rectangle (0,lx) x (0,ly)."""
    X, Y = np.meshgrid(np.linspace(0.0, lx, nx + 1),
                       np.linspace(0.0, ly, ny + 1), indexing="ij")
    return structured_tri_mesh(X, Y)


def quarter_annulus_mesh(a, R, nr, nt):
    """Graded polar triangulation of {a <= r <= R, 0 <= theta <= pi/2}.

    The radial nodes are geometric,  r_i = a (R/a)^(i/nr), which keeps the cells
    close to square all the way in to the hole - essential for resolving the
    1/r^2 stress concentration that the topological derivative measures.
    Logical index i runs radially, j runs in theta.
    """
    r = a * (R / a) ** (np.linspace(0.0, 1.0, nr + 1))
    th = np.linspace(0.0, 0.5 * np.pi, nt + 1)
    Rg, Tg = np.meshgrid(r, th, indexing="ij")
    return structured_tri_mesh(Rg * np.cos(Tg), Rg * np.sin(Tg))


# -----------------------------------------------------------------------------
# 3. CST elasticity operator
# -----------------------------------------------------------------------------
class Elasticity2D:
    """Assemble/solve  -div sigma(u) = 0  with P1 triangles.

    The element stiffness is *linear* in the element Young modulus,
        K_e(E_e) = E_e * K0_e ,        K0_e = A_e B_e^T D(1, nu) B_e ,
    so the unit-modulus element matrices `K0` are built once and every
    re-assembly during the optimization is a single scaling of the value array.
    """

    def __init__(self, nodes, tris, nu=0.3, model="plane_stress"):
        self.nodes, self.tris = np.asarray(nodes, float), np.asarray(tris, int)
        self.nu, self.model = nu, model
        self.n_nodes = len(self.nodes)
        self.n_tris = len(self.tris)
        self.ndof = 2 * self.n_nodes
        self.D1 = hooke_2d(1.0, nu, model)          # unit Young modulus

        x, y = self.nodes[self.tris, 0], self.nodes[self.tris, 1]
        b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], -1)
        c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], -1)
        two_a = x[:, 0] * b[:, 0] + x[:, 1] * b[:, 1] + x[:, 2] * b[:, 2]
        self.area = 0.5 * two_a
        if np.any(self.area <= 0):
            raise ValueError("degenerate or clockwise triangle in the mesh")

        # Strain-displacement operator B (T, 3, 6): eps_voigt = B u_e.
        B = np.zeros((self.n_tris, 3, 6))
        B[:, 0, 0::2] = b / two_a[:, None]
        B[:, 1, 1::2] = c / two_a[:, None]
        B[:, 2, 0::2] = c / two_a[:, None]
        B[:, 2, 1::2] = b / two_a[:, None]
        self.B = B
        self.K0 = self.area[:, None, None] * np.einsum("tki,kl,tlj->tij", B, self.D1, B)

        self.edof = (2 * self.tris[:, :, None] + np.array([0, 1])).reshape(-1, 6)
        self._rows = np.repeat(self.edof[:, :, None], 6, axis=2).ravel()
        self._cols = np.repeat(self.edof[:, None, :], 6, axis=1).ravel()

    # -- assembly -------------------------------------------------------------
    def stiffness(self, E):
        """Global stiffness for per-triangle Young moduli E (T,)."""
        vals = (np.asarray(E, float)[:, None, None] * self.K0).ravel()
        K = sp.coo_matrix((vals, (self._rows, self._cols)),
                          shape=(self.ndof, self.ndof)).tocsr()
        return K

    # -- boundary conditions --------------------------------------------------
    def edge_load(self, edges, traction):
        """Consistent P1 load vector for a constant traction on straight edges.

        `edges` is (M, 2) node pairs, `traction` is (M, 2) or (2,).  For a P1
        edge of length l the exact integral of  t . N_i  is  l t / 2  at each
        endpoint.
        """
        edges = np.asarray(edges, int)
        t = np.broadcast_to(np.asarray(traction, float), (len(edges), 2))
        p = self.nodes[edges]                                   # (M, 2, 2)
        length = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
        f = np.zeros(self.ndof)
        contrib = 0.5 * length[:, None] * t                     # (M, 2)
        for k in range(2):
            np.add.at(f, 2 * edges[:, k], contrib[:, 0])
            np.add.at(f, 2 * edges[:, k] + 1, contrib[:, 1])
        return f

    def solve(self, E, f, fixed):
        """Solve K(E) u = f with u = 0 on the dof set `fixed`. Returns (u, J)."""
        K = self.stiffness(E)
        free = np.ones(self.ndof, bool)
        free[np.asarray(fixed, int)] = False
        u = np.zeros(self.ndof)
        u[free] = spla.spsolve(K[free][:, free].tocsc(), f[free])
        return u, float(f @ u)

    # -- postprocessing -------------------------------------------------------
    def strain(self, u):
        """Element strains (T, 3) in Voigt form (xx, yy, gamma_xy)."""
        return np.einsum("tij,tj->ti", self.B, u[self.edof])

    def stress(self, u, E):
        """Element stresses (T, 3).  `E` may be per-triangle or a scalar."""
        E = np.broadcast_to(np.asarray(E, float), (self.n_tris,))
        return E[:, None] * (self.strain(u) @ self.D1.T)


# -----------------------------------------------------------------------------
# 4. The topological derivative of the compliance
# -----------------------------------------------------------------------------
def topological_derivative(sigma, E, nu, model="plane_stress"):
    """D_T J for a traction-free circular hole, per unit hole area.

    With the normalization  J(Omega \\ B_eps) = J(Omega) + pi eps^2 D_T J + o(eps^2),

        D_T J = k(E, nu) * [ 4 sigma:sigma - (tr sigma)^2 ],
        k = 1/E                (plane stress)
        k = (1 - nu^2)/E       (plane strain)

    `sigma` is (T, 3) in Voigt form.  The bracket is a positive definite form:
    in principal stresses it equals 3(s1^2 + s2^2) - 2 s1 s2, so punching a hole
    can never lower the compliance.
    """
    sigma = np.atleast_2d(np.asarray(sigma, float))
    s2 = sigma[:, 0] ** 2 + sigma[:, 1] ** 2 + 2.0 * sigma[:, 2] ** 2
    tr = sigma[:, 0] + sigma[:, 1]
    k = 1.0 / E if model == "plane_stress" else (1.0 - nu ** 2) / E
    return k * (4.0 * s2 - tr ** 2)
