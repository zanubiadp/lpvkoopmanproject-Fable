"""Variable-projection (VARPRO) machinery for Koopman eigenvalue learning.

The data-driven construction of Korda & Mezic (2020), Sec. IV, reduces to a
classical separable least-squares problem: for one state component h and
candidate eigenvalues Lambda, the boundary values enter linearly,

    min_{g}  sum_j || h^(j) - V(Lambda) g^(j) ||^2,

where h^(j) is the component sampled along trajectory j and
V(Lambda)[k, m] = exp(lambda_m * t_k) is shared by all trajectories (uniform
sampling). Eliminating g gives the projected ("variable projection") cost

    J(Lambda) = sum_j || (I - P_V) h^(j) ||^2,                 (paper eq. 39)

a smooth function of Lambda with a cheap exact gradient (paper eq. 40):

    dJ/dtheta = -2 * trace( R^T (dV/dtheta) G ),
    R = H - V G,  G = V^+ H.

We work in the real conjugate-pair parametrization (thesis Appendix A): a
pair alpha +- i beta contributes the two real columns

    exp(alpha t) cos(beta t),   exp(alpha t) sin(beta t),

and a real eigenvalue contributes exp(alpha t). For real data this spans the
same space as the complex pair {e^{lambda t}, e^{conj(lambda) t}}, so the
projected cost is identical (verified in tests), while keeping all arithmetic
real and the eigenvalue set conjugate-closed by construction.

Column order convention (must match model assembly): pair p occupies columns
(2p, 2p+1) = (cos, sin); real eigenvalues follow, one column each.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .spectrum import Spectrum

Array = np.ndarray


def basis_matrix(spec: Spectrum, t: Array) -> Array:
    """Real exponential/damped-oscillation basis V of shape (len(t), spec.size)."""
    t = np.asarray(t, dtype=float).reshape(-1)
    cols: list[Array] = []
    for a, b in spec.pairs:
        ea = np.exp(a * t)
        cols.append(ea * np.cos(b * t))
        cols.append(ea * np.sin(b * t))
    for r in spec.reals:
        cols.append(np.exp(r * t))
    if not cols:
        return np.empty((len(t), 0))
    return np.column_stack(cols)


def complex_basis_matrix(lams: Array, t: Array) -> Array:
    """Complex basis V[k, m] = exp(lams[m] * t_k) (for equivalence checks)."""
    t = np.asarray(t, dtype=float).reshape(-1)
    lams = np.asarray(lams, dtype=complex).reshape(-1)
    return np.exp(np.outer(t, lams))


def fit_boundary(V: Array, H: Array, rcond: float = 1e-12) -> tuple[Array, Array]:
    """Least-squares boundary coefficients for all trajectories at once.

    Solves min_G || H - V G ||_F with column equilibration for conditioning
    (the lattice columns exp(alpha t) span many orders of magnitude).

    Returns (G, R): coefficients (N, M_t) and residual matrix R = H - V G.
    """
    if V.shape[1] == 0:
        return np.empty((0, H.shape[1])), H.copy()
    scale = np.linalg.norm(V, axis=0)
    scale[scale == 0] = 1.0
    Gs, *_ = np.linalg.lstsq(V / scale, H, rcond=rcond)
    G = Gs / scale[:, None]
    R = H - V @ G
    return G, R


def projection_cost(spec: Spectrum, t: Array, H: Array) -> float:
    """J = sum of squared residuals after projecting H onto the basis."""
    V = basis_matrix(spec, t)
    _, R = fit_boundary(V, H)
    return float(np.sum(R * R))


def projection_cost_grad(
    theta: Array, n_pairs: int, t: Array, H: Array
) -> tuple[float, Array]:
    """Projected cost and its exact gradient w.r.t. packed real parameters.

    theta layout: [a_1, b_1, ..., a_P, b_P, r_1, ..., r_R].
    """
    spec = Spectrum.unpack(theta, n_pairs)
    t = np.asarray(t, dtype=float).reshape(-1)
    V = basis_matrix(spec, t)
    G, R = fit_boundary(V, H)

    J = float(np.sum(R * R))
    grad = np.zeros_like(np.asarray(theta, dtype=float))

    # dJ/dtheta = -2 tr(R^T dV G); dV/dtheta touches at most two columns.
    tcol = t[:, None]
    for p in range(n_pairs):
        c, s = V[:, 2 * p], V[:, 2 * p + 1]
        # d/d alpha: columns (t*cos, t*sin)
        dV_alpha = tcol * V[:, 2 * p : 2 * p + 2]
        grad[2 * p] = -2.0 * np.sum((R.T @ dV_alpha).T * G[2 * p : 2 * p + 2])
        # d/d beta: (-t*sin, t*cos)
        dV_beta = np.column_stack([-t * s, t * c])
        grad[2 * p + 1] = -2.0 * np.sum((R.T @ dV_beta).T * G[2 * p : 2 * p + 2])
    for j in range(spec.n_reals):
        col = 2 * n_pairs + j
        dv = t * V[:, col]
        grad[col] = -2.0 * float((R.T @ dv) @ G[col])
    return J, grad


@dataclass
class RefineResult:
    spectrum: Spectrum
    cost: float
    cost_initial: float
    n_iter: int
    converged: bool
    message: str


def refine_spectrum(
    spec0: Spectrum,
    t: Array,
    H: Array,
    alpha_bounds: tuple[float, float] = (-20.0, 20.0),
    beta_max: float | None = None,
    maxiter: int = 1000,
    n_restarts: int = 0,
    perturb_scale: float = 0.1,
    rng: np.random.Generator | None = None,
) -> RefineResult:
    """Refine eigenvalues by L-BFGS-B on the projected cost, starting from
    ``spec0`` (typically the DMD lattice).

    This replaces (a) fmincon in the reference MATLAB implementation and
    (b) the CMA-ES global search of the previous Python port. The cost
    landscape is multimodal, but with the lattice initialization a local
    quasi-Newton method converges in seconds; optional restarts perturb the
    initial point to guard against poor local minima.
    """
    t = np.asarray(t, dtype=float).reshape(-1)
    if beta_max is None:
        ts = float(t[1] - t[0])
        beta_max = np.pi / ts  # Nyquist limit for the sampling rate
    n_pairs = spec0.n_pairs

    lo, hi = alpha_bounds
    bounds = []
    for _ in range(n_pairs):
        bounds += [(lo, hi), (1e-9, beta_max)]
    bounds += [(lo, hi)] * spec0.n_reals

    theta0 = spec0.pack()
    J0 = projection_cost(spec0, t, H)

    starts = [theta0]
    if n_restarts > 0:
        rng = rng or np.random.default_rng(0)
        scale = perturb_scale * max(1.0, float(np.abs(theta0).max()))
        for _ in range(n_restarts):
            starts.append(theta0 + scale * rng.standard_normal(theta0.shape))

    best: RefineResult | None = None
    for th0 in starts:
        res = minimize(
            projection_cost_grad,
            np.clip(th0, [b[0] for b in bounds], [b[1] for b in bounds]),
            args=(n_pairs, t, H),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-12},
        )
        cand = RefineResult(
            spectrum=Spectrum.unpack(res.x, n_pairs),
            cost=float(res.fun),
            cost_initial=J0,
            n_iter=int(res.nit),
            converged=bool(res.success),
            message=str(res.message),
        )
        if best is None or cand.cost < best.cost:
            best = cand
    assert best is not None
    return best
