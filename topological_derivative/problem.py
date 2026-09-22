"""
The design problem: geometry, boundary conditions, and how to draw them.

`Problem` carries everything the solver needs - the mesh, the elasticity
operator, the load vector, the constrained dofs, the permanently void cells and
the protected "keep" cells - together with the handful of things that are
specific to a load case but only matter to the figures: the view limits, the
boundary-condition symbols and the titles.

Each load case in `examples/` subclasses it, declaring its geometry and schedule
as class attributes and implementing `boundary_conditions` (which runs once from
`__init__`, with the mesh and the elasticity operator already built) plus its
own `draw_supports` / `draw_load`.  Because every load case answers for itself,
neither the solver nor the plotting code has to branch on which problem it is
looking at.
"""

import numpy as np

import fem

E_SOLID, E_MIN, NU = 1.0, 1e-6, 0.3
MODEL = "plane_stress"


class Problem:
    """One compliance-minimization load case on a triangulated rectangle."""

    # -- identity, declared by each load case ---------------------------------
    name = ""
    title = ""                  # human-readable name, for the animation header
    label = ""                  # one-line description of the boundary data
    support_condition = ""      # prose for td_results.json; defaults to `label`
    cache_tag = ""              # keeps cached runs of related variants apart

    # -- geometry and schedule ------------------------------------------------
    lx, ly = 1.0, 1.0
    shape = (100, 100)          # default cell counts (nx, ny)
    vol_frac = 0.40             # target material fraction of the design domain
    rmin_cells = 3.5            # cone-filter radius, in element sizes
    evol_rate = 0.02            # per-iteration volume reduction
    n_iter = 90                 # updates, i.e. solved states minus one

    # -- figure geometry ------------------------------------------------------
    ylim = (-0.09, 1.09)        # y view limits, in units of ly
    figsize = (8.4, 4.2)
    figure_file = None          # set for a standalone design figure
    gif_file = None             # set for an animated load case

    def __init__(self, nx=None, ny=None, vol_frac=None, rmin_cells=None):
        self.nx = self.shape[0] if nx is None else nx
        self.ny = self.shape[1] if ny is None else ny
        if vol_frac is not None:
            self.vol_frac = vol_frac
        if rmin_cells is not None:
            self.rmin_cells = rmin_cells

        self.nodes, self.tris, self.cell, _ = fem.rect_mesh(
            self.lx, self.ly, self.nx, self.ny)
        self.pb = fem.Elasticity2D(self.nodes, self.tris, nu=NU, model=MODEL)

        self.f = np.zeros(self.pb.ndof)         # consistent nodal load
        self.fixed = np.zeros(0, int)           # constrained dofs
        self.void = np.zeros(self.n_cells, bool)    # outside the design domain
        self.keep = np.zeros(self.n_cells, bool)    # protected solid
        self.boundary_conditions()

    def __repr__(self):
        return (f"{type(self).__name__}({self.nx}x{self.ny}, "
                f"V={self.vol_frac}, rmin={self.rmin_cells})")

    # -- geometry helpers -----------------------------------------------------
    @property
    def n_cells(self):
        return self.nx * self.ny

    @property
    def h(self):
        """Cell size (hx, hy).  Uniform, so volume fractions are cell counts."""
        return np.array([self.lx / self.nx, self.ly / self.ny])

    def cell_centers(self):
        hx, hy = self.h
        xc = (np.arange(self.nx) + 0.5) * hx
        yc = (np.arange(self.ny) + 0.5) * hy
        return np.meshgrid(xc, yc, indexing="ij")

    def node_grid(self):
        """Node ids on the logical grid, shape (nx + 1, ny + 1)."""
        return np.arange((self.nx + 1) * (self.ny + 1)).reshape(self.nx + 1,
                                                                self.ny + 1)

    def node_column(self, x):
        """Grid column index of the node line nearest to abscissa `x`."""
        return np.rint(np.asarray(x, float) / self.h[0]).astype(int)

    def node_row(self, y):
        """Grid row index of the node line nearest to ordinate `y`."""
        return np.rint(np.asarray(y, float) / self.h[1]).astype(int)

    # -- boundary-condition helpers -------------------------------------------
    def boundary_conditions(self):
        """Fill in `f`, `fixed` and, where they are not empty, `void`/`keep`."""
        raise NotImplementedError

    @staticmethod
    def clamp(node_ids):
        """Dofs constraining both displacement components at `node_ids`."""
        nid = np.asarray(node_ids, int).ravel()
        return np.concatenate([2 * nid, 2 * nid + 1])

    @staticmethod
    def edges_on_line(line, mask=None):
        """Node-pair edges along a boundary grid line, kept where `mask` holds."""
        pairs = np.column_stack([line[:-1], line[1:]])
        return pairs if mask is None else pairs[mask[:-1] & mask[1:]]

    @property
    def fixed_nodes(self):
        """Nodes with at least one constrained displacement component."""
        return np.unique(np.asarray(self.fixed, int) // 2)

    @property
    def loaded_nodes(self):
        """Nodes carrying a nonzero nodal force."""
        return np.unique(np.where(np.abs(self.f) > 0)[0] // 2)

    # -- boundary-condition symbols -------------------------------------------
    def annotate(self, ax):
        """Draw the displacement constraints and the prescribed traction.

        The two are drawn separately because they are different objects: the
        bridges have pinned *point* supports rather than clamped edges, and
        those point constraints belong to the discrete benchmark, while
        Gamma_D labels the constrained components.  Both are labelled by the
        symbol alone, small enough for a sweep panel; what they stand for is
        the post's job to say.
        """
        self.draw_supports(ax)
        self.draw_load(ax)

    def draw_supports(self, ax):
        """Draw Gamma_D: the displacement constraints and their label."""
        raise NotImplementedError

    def draw_load(self, ax):
        """Draw Gamma_N: the loaded boundary, its traction and its label."""
        raise NotImplementedError
