"""Gradients of the learned eigenfunctions (the lift map Phi).

The LPV input matrix is B(x) = (dPhi/dx) g_c(x) (Iacob et al. 2024), so the
control extension stands or falls with the quality of dPhi/dx. Phi is known
as tabulated values on the training trajectory cloud; two estimators are
provided:

  * "mls" (default): moving least squares - fit a local weighted quadratic
    model of every lifted coordinate on the k nearest training samples and
    read off its analytic gradient. Mesh-free, no step-size parameter, one
    shared (k x 6) design per query for all N coordinates.
  * "fd": central finite differences of the interpolated lift (the thesis
    used this with order 2/4 stencils on a linear interpolant; provided for
    cross-checking and to quantify its step-size sensitivity).

Quality diagnostic: along the autonomous flow the tabulated lift satisfies
zdot = A z exactly by construction, so the PDE residual

    r(x) = (dPhi/dx)(x) f(x) - A Phi(x)

vanishes at the data up to interpolation/gradient error. ``pde_residual``
evaluates its normalized magnitude anywhere, giving a direct, ground-truth
measure of gradient quality (used in tests and the validation report).
"""
from __future__ import annotations

import numpy as np

from .model import KoopmanEigenModel

Array = np.ndarray


def lift_jacobian_mls(
    model: KoopmanEigenModel,
    x: Array,
    k_neighbors: int = 40,
    degree: int = 2,
) -> Array:
    """Moving-least-squares Jacobian of the lift at query points.

    x : (n,) or (Q, n). Returns (N, n) or (Q, N, n).

    For each query, the k nearest training samples are fitted with a
    weighted polynomial model (degree 1 or 2) per lifted coordinate; the
    gradient is the model's linear term evaluated at the query point.
    Weights: Wendland-type (1 - (d/d_max)^3)^3.
    """
    x = np.asarray(x, dtype=float)
    single = x.ndim == 1
    Q = np.atleast_2d(x)
    n = Q.shape[1]
    if n != 2:
        raise NotImplementedError("MLS gradients currently assume 2-D state")

    dist, idx = model._tree.query(Q, k=k_neighbors)
    out = np.empty((len(Q), model.N, n))
    for q in range(len(Q)):
        P = model.points[idx[q]] - Q[q]  # (k, 2) offsets
        scale = max(dist[q].max(), 1e-12)
        Ps = P / scale
        d = dist[q] / scale
        w = (1.0 - np.minimum(d, 1.0) ** 3) ** 3 + 1e-6
        cols = [np.ones(len(Ps)), Ps[:, 0], Ps[:, 1]]
        if degree >= 2:
            cols += [Ps[:, 0] ** 2, Ps[:, 0] * Ps[:, 1], Ps[:, 1] ** 2]
        Dmat = np.column_stack(cols)
        sw = np.sqrt(w)[:, None]
        Zn = model.lifted_at_points[idx[q]]  # (k, N)
        coef, *_ = np.linalg.lstsq(sw * Dmat, sw * Zn, rcond=None)
        out[q] = coef[1:3].T / scale  # d/dx of the local model at offset 0
    return out[0] if single else out


def lift_jacobian_fd(
    model: KoopmanEigenModel,
    x: Array,
    h: float = 1e-3,
    order: int = 4,
) -> Array:
    """Finite-difference Jacobian of the *interpolated* lift.

    Mirrors the reference implementation's approach (build_B_numgrad.m used
    order-2/4 central stencils; note it defaulted to h = 0.1, far too coarse
    for features of size ~0.05 - measured in the validation report).
    """
    x = np.asarray(x, dtype=float)
    single = x.ndim == 1
    Q = np.atleast_2d(x)
    n = Q.shape[1]
    out = np.empty((len(Q), model.N, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = h
        if order == 2:
            out[:, :, j] = (model.lift(Q + e) - model.lift(Q - e)) / (2 * h)
        elif order == 4:
            out[:, :, j] = (
                -model.lift(Q + 2 * e)
                + 8 * model.lift(Q + e)
                - 8 * model.lift(Q - e)
                + model.lift(Q - 2 * e)
            ) / (12 * h)
        else:
            raise ValueError("order must be 2 or 4")
    return out[0] if single else out


def lift_jacobian(
    model: KoopmanEigenModel, x: Array, method: str = "mls", **kw
) -> Array:
    if method == "mls":
        return lift_jacobian_mls(model, x, **kw)
    if method == "fd":
        return lift_jacobian_fd(model, x, **kw)
    raise ValueError(f"unknown gradient method '{method}'")


def pde_residual(
    model: KoopmanEigenModel,
    f,
    x: Array,
    method: str = "mls",
    **kw,
) -> Array:
    """Normalized eigenfunction-PDE residual ||J f - A z|| / ||A z|| at x.

    f : autonomous vector field callable, x : (Q, n). Returns (Q,).
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    J = lift_jacobian(model, x, method=method, **kw)  # (Q, N, n)
    Z = model.lift(x)  # (Q, N)
    A = model.A_continuous()
    Az = Z @ A.T
    fx = np.stack([np.asarray(f(xi), dtype=float) for xi in x])  # (Q, n)
    Jf = np.einsum("qij,qj->qi", J, fx)
    return np.linalg.norm(Jf - Az, axis=1) / np.maximum(
        np.linalg.norm(Az, axis=1), 1e-12
    )
