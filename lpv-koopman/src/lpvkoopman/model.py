"""Assembled Koopman eigenfunction model: real lifted linear dynamics,
eigenfunction interpolation, and closed-form prediction.

Structure (all real):

  - Each state component i has its own conjugate-closed spectrum (Korda &
    Mezic split the eigenfunction budget per predicted output). A conjugate
    pair alpha +- i beta contributes a 2-dimensional lifted block evolving as

        zdot = [[alpha, beta], [-beta, alpha]] z,

    whose first coordinate carries the reconstruction; a real eigenvalue
    contributes a scalar block zdot = alpha z.

  - x_hat_i = sum of the carrier coordinates of component i's blocks
    (paper eq. (41): C = block-diagonal of all-ones rows). An optional
    least-squares refit of C on interpolated lifted values (paper eq. (42))
    is provided as ``refit_C``.

  - Off-data evaluation of the lifted state ("lifting" a new initial
    condition) interpolates the tabulated eigenfunction values on the
    training cloud: piecewise-linear on a shared Delaunay triangulation by
    default (as in the paper), optionally Clough-Tocher C1 cubic. Points
    outside the convex hull fall back to nearest-neighbor values.

Prediction is the exact matrix exponential of the block structure - no ODE
integration error on the lifted side, and predictions are real by
construction (the goal of the thesis Appendix-A reformulation).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator
from scipy.spatial import Delaunay, cKDTree

from .spectrum import Spectrum
from .varpro import basis_matrix

Array = np.ndarray


@dataclass
class ComponentFit:
    """Per-state-component fit: spectrum + boundary coefficients.

    G has shape (spec.size, M_t); rows follow the basis column convention
    (pair p -> rows 2p, 2p+1 = cos, sin coefficients; then reals).
    """

    spec: Spectrum
    G: Array
    cost: float = np.nan
    info: dict = field(default_factory=dict)


def lifted_values(spec: Spectrum, t: Array, G: Array) -> Array:
    """Tabulate all lifted coordinates along the training trajectories.

    Returns Z of shape (spec.size, M_s + 1, M_t): for pair p the carrier
    coordinate z1 = a e^{at}cos(bt) + b e^{at}sin(bt) and its quadrature
    z2 = -a e^{at}sin(bt) + b e^{at}cos(bt); for real eigenvalues z = e^{at} g.
    """
    t = np.asarray(t, dtype=float).reshape(-1)
    V = basis_matrix(spec, t)  # (T, N)
    n_t, m_t = len(t), G.shape[1]
    Z = np.empty((spec.size, n_t, m_t))
    for p in range(spec.n_pairs):
        c, s = V[:, 2 * p], V[:, 2 * p + 1]
        a = G[2 * p, :]  # (M_t,)
        b = G[2 * p + 1, :]
        Z[2 * p] = c[:, None] * a[None, :] + s[:, None] * b[None, :]
        Z[2 * p + 1] = -s[:, None] * a[None, :] + c[:, None] * b[None, :]
    off = 2 * spec.n_pairs
    for j in range(spec.n_reals):
        Z[off + j] = V[:, off + j][:, None] * G[off + j, None, :]
    return Z


class KoopmanEigenModel:
    """Real linear lifted predictor built from learned eigenfunctions."""

    def __init__(
        self,
        components: list[ComponentFit],
        points: Array,
        lifted_at_points: Array,
        interpolation: str = "linear",
    ):
        """
        Parameters
        ----------
        components:
            One ComponentFit per predicted state component (in order).
        points:
            (P, n) training sample locations (the trajectory cloud).
        lifted_at_points:
            (P, N_lift) tabulated lifted coordinates at ``points`` (all
            components concatenated in block order).
        interpolation:
            "linear" (paper's choice) or "cubic" (Clough-Tocher C1).
        """
        self.components = components
        self.n_states = len(components)
        self.block_sizes = [c.spec.size for c in components]
        self.N = int(sum(self.block_sizes))
        if lifted_at_points.shape[1] != self.N:
            raise ValueError("lifted value table does not match lift dimension")

        self.points = points
        self.lifted_at_points = lifted_at_points
        self._tree = cKDTree(points)
        self._tri = Delaunay(points)
        if interpolation == "linear":
            self._interp = LinearNDInterpolator(self._tri, lifted_at_points)
        elif interpolation == "cubic":
            self._interp = CloughTocher2DInterpolator(self._tri, lifted_at_points)
        else:
            raise ValueError(f"unknown interpolation '{interpolation}'")
        self.interpolation = interpolation
        self.C = self._default_C()
        self.last_lift_fallbacks = 0

    # --- matrices ------------------------------------------------------
    def _default_C(self) -> Array:
        """Paper eq. (41): each state sums the carrier coordinates of its blocks."""
        C = np.zeros((self.n_states, self.N))
        col = 0
        for i, comp in enumerate(self.components):
            for _ in range(comp.spec.n_pairs):
                C[i, col] = 1.0  # carrier (cos) coordinate
                col += 2
            for _ in range(comp.spec.n_reals):
                C[i, col] = 1.0
                col += 1
        return C

    def A_continuous(self) -> Array:
        """Real block-diagonal continuous-time lifted dynamics matrix."""
        A = np.zeros((self.N, self.N))
        col = 0
        for comp in self.components:
            for a, b in comp.spec.pairs:
                A[col, col] = a
                A[col, col + 1] = b
                A[col + 1, col] = -b
                A[col + 1, col + 1] = a
                col += 2
            for r in comp.spec.reals:
                A[col, col] = r
                col += 1
        return A

    def eigenvalues(self) -> Array:
        """All continuous-time eigenvalues (complex), per component order."""
        return np.concatenate([c.spec.to_complex() for c in self.components])

    # --- lifting ---------------------------------------------------------
    def lift(self, x: Array) -> Array:
        """Evaluate the lifted state at arbitrary points by interpolation.

        x : (n,) or (Q, n). Returns (N,) or (Q, N). Points outside the hull
        of the training cloud use nearest-neighbor values; the count of such
        fallbacks is stored in ``last_lift_fallbacks``.
        """
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        Q = np.atleast_2d(x)
        Z = self._interp(Q)
        bad = np.isnan(Z).any(axis=1)
        self.last_lift_fallbacks = int(bad.sum())
        if bad.any():
            _, idx = self._tree.query(Q[bad])
            Z[bad] = self.lifted_at_points[idx]
        return Z[0] if single else Z

    # --- prediction -----------------------------------------------------
    def propagate(self, z0: Array, t: Array) -> Array:
        """Closed-form propagation of the lifted state: Z(t) = e^{A t} z0.

        z0 : (N,), t : (T,). Returns (T, N). Uses the per-block rotation /
        scaling form of the matrix exponential (exact, no integration error).
        """
        t = np.asarray(t, dtype=float).reshape(-1)
        Z = np.empty((len(t), self.N))
        col = 0
        for comp in self.components:
            for a, b in comp.spec.pairs:
                ea = np.exp(a * t)
                cb, sb = np.cos(b * t), np.sin(b * t)
                z1, z2 = z0[col], z0[col + 1]
                Z[:, col] = ea * (z1 * cb + z2 * sb)
                Z[:, col + 1] = ea * (-z1 * sb + z2 * cb)
                col += 2
            for r in comp.spec.reals:
                Z[:, col] = np.exp(r * t) * z0[col]
                col += 1
        return Z

    def predict(self, x0: Array, t: Array) -> Array:
        """Predict the state trajectory from a new initial condition.

        Returns (T, n): x_hat(t_k) = C exp(A t_k) Phi(x0).
        """
        z0 = self.lift(np.asarray(x0, dtype=float))
        return self.propagate(z0, t) @ self.C.T

    # --- optional C refit (paper eq. 42) ----------------------------------
    def refit_C(
        self,
        points: Array,
        targets: Array,
        ridge: float = 0.0,
    ) -> Array:
        """Refit C by (ridge) least squares on interpolated lifted values.

        points : (Q, n) sample locations, targets : (Q, n) state values to
        reproduce (usually targets == points for h(x) = x). Updates and
        returns self.C.
        """
        Z = self.lift(points)  # (Q, N)
        A = Z.T @ Z + ridge * np.eye(self.N)
        B = Z.T @ targets
        self.C = np.linalg.solve(A, B).T
        return self.C
