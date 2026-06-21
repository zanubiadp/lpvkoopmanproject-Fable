"""Continuous-time dynamical systems and trajectory data generation.

Only autonomous dynamics are needed for the current milestone. The
``ContinuousSystem`` interface keeps a hook (``input_matrix``) for the future
LPV input-matrix construction B(x) = (dPhi/dx) g_c(x), which is out of scope
here but drives the architecture.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp


Array = np.ndarray


@dataclass
class ContinuousSystem:
    """Autonomous continuous-time system  xdot = f(x)  with vectorized f.

    Attributes
    ----------
    f:
        Right-hand side. Accepts state of shape (n,) or (n, m) and returns
        the same shape (vectorized over columns).
    n:
        State dimension.
    name:
        Human-readable identifier.
    g:
        Optional control vector field for the future control-affine extension
        xdot = f(x) + g(x) u. Unused in the autonomous pipeline.
    """

    f: Callable[[Array], Array]
    n: int
    name: str = "system"
    g: Callable[[Array], Array] | None = None

    def simulate(
        self,
        x0: Array,
        t_eval: Array,
        rtol: float = 1e-10,
        atol: float = 1e-12,
        method: str = "RK45",
    ) -> Array:
        """Integrate from ``x0`` and sample at ``t_eval``.

        Returns array of shape (len(t_eval), n).
        """
        x0 = np.asarray(x0, dtype=float).reshape(self.n)
        sol = solve_ivp(
            lambda _t, x: self.f(x),
            (float(t_eval[0]), float(t_eval[-1])),
            x0,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            method=method,
        )
        if not sol.success:
            raise RuntimeError(f"Integration failed for x0={x0}: {sol.message}")
        return sol.y.T

    def simulate_many(
        self,
        x0s: Array,
        t_eval: Array,
        rtol: float = 1e-10,
        atol: float = 1e-12,
    ) -> Array:
        """Integrate a batch of initial conditions.

        Parameters
        ----------
        x0s : (M, n) initial conditions.

        Returns
        -------
        (M, len(t_eval), n) trajectory tensor.
        """
        x0s = np.atleast_2d(np.asarray(x0s, dtype=float))
        out = np.empty((x0s.shape[0], len(t_eval), self.n))
        for j, x0 in enumerate(x0s):
            out[j] = self.simulate(x0, t_eval, rtol=rtol, atol=atol)
        return out

    def jacobian_fd(self, x: Array, eps: float = 1e-7) -> Array:
        """Central finite-difference Jacobian of f at x."""
        x = np.asarray(x, dtype=float).reshape(self.n)
        J = np.empty((self.n, self.n))
        for k in range(self.n):
            dx = np.zeros(self.n)
            dx[k] = eps
            J[:, k] = (self.f(x + dx) - self.f(x - dx)) / (2 * eps)
        return J


def vdp_scaled() -> ContinuousSystem:
    """Scaled Van der Pol oscillator used by Korda & Mezic (2020), Sec. VI-A.

        x1dot = 2 x2
        x2dot = -0.8 x1 + 2 x2 - 10 x1^2 x2

    The origin is an unstable focus (eigenvalues 1 +- i sqrt(0.6)); the
    attracting limit cycle has amplitude ~0.5 in x1. All published benchmark
    numbers in the paper (Table I) refer to this scaling.
    """

    def f(x: Array) -> Array:
        x = np.asarray(x, dtype=float)
        x1, x2 = x[0], x[1]
        return np.stack([2.0 * x2, -0.8 * x1 + 2.0 * x2 - 10.0 * (x1**2) * x2])

    def g(x: Array) -> Array:
        x = np.asarray(x, dtype=float)
        shape = (2,) if x.ndim == 1 else (2, x.shape[1])
        out = np.zeros(shape)
        out[1] = 1.0
        return out

    return ContinuousSystem(f=f, n=2, name="vdp_scaled", g=g)


def vdp_classic(mu: float = 1.0) -> ContinuousSystem:
    """Textbook Van der Pol oscillator x1dot = x2, x2dot = mu(1-x1^2)x2 - x1."""

    def f(x: Array) -> Array:
        x = np.asarray(x, dtype=float)
        x1, x2 = x[0], x[1]
        return np.stack([x2, mu * (1.0 - x1**2) * x2 - x1])

    return ContinuousSystem(f=f, n=2, name=f"vdp_classic_mu{mu:g}")


def linear_system(A: Array) -> ContinuousSystem:
    """Linear system xdot = A x (used in tests: exact Koopman spectrum)."""
    A = np.asarray(A, dtype=float)

    def f(x: Array) -> Array:
        return A @ np.asarray(x, dtype=float)

    return ContinuousSystem(f=f, n=A.shape[0], name="linear")


@dataclass
class TrajectoryData:
    """Uniformly sampled bundle of autonomous trajectories.

    X : (M_t, M_s + 1, n)  trajectory tensor
    t : (M_s + 1,)         shared sample times, t[k] = k * ts
    """

    X: Array
    t: Array
    meta: dict = field(default_factory=dict)

    @property
    def n_traj(self) -> int:
        return self.X.shape[0]

    @property
    def n_samples(self) -> int:
        return self.X.shape[1]

    @property
    def n_states(self) -> int:
        return self.X.shape[2]

    @property
    def ts(self) -> float:
        return float(self.t[1] - self.t[0])

    def component(self, i: int) -> Array:
        """Stacked component matrix H_i of shape (M_s + 1, M_t).

        Column j holds state component i along trajectory j; this is the
        right-hand side of the per-trajectory least-squares fits.
        """
        return self.X[:, :, i].T.copy()

    def flat_points(self) -> Array:
        """All sample points as an (M_t * (M_s+1), n) array."""
        return self.X.reshape(-1, self.n_states)


def circle_initial_conditions(
    radius: float, count: int, center: tuple[float, float] = (0.0, 0.0)
) -> Array:
    """Equally spaced initial conditions on a circle (Korda & Mezic VdP setup)."""
    th = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.column_stack(
        [center[0] + radius * np.cos(th), center[1] + radius * np.sin(th)]
    )


def generate_data(
    system: ContinuousSystem,
    x0s: Array,
    horizon: float,
    ts: float,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    backward: float = 0.0,
) -> TrajectoryData:
    """Simulate the system from ``x0s`` for ``horizon`` seconds at step ``ts``.

    If ``backward`` > 0, each trajectory is additionally integrated backward
    in time for that long, producing samples at t in [-backward, horizon]
    with the initial condition in the middle. The eigenfunction construction
    is two-sided (phi(S_t x) = e^{lambda t} phi(x) for negative t as well),
    so the fitting machinery needs no changes; negative times simply appear
    in the exponential basis. For systems with an unstable focus this covers
    the deep interior (backward flow contracts toward the fixed point) at
    a much smaller time-depth than forward-only data from a tiny circle.
    """
    n_fwd = int(round(horizon / ts))
    t_fwd = np.arange(n_fwd + 1) * ts
    X_fwd = system.simulate_many(x0s, t_fwd, rtol=rtol, atol=atol)
    if backward <= 0:
        return TrajectoryData(
            X=X_fwd, t=t_fwd, meta={"system": system.name, "horizon": horizon, "ts": ts}
        )

    n_bwd = int(round(backward / ts))
    t_bwd = np.arange(n_bwd + 1) * ts
    rev = ContinuousSystem(f=lambda x: -system.f(x), n=system.n, name=system.name)
    X_bwd = rev.simulate_many(np.atleast_2d(x0s), t_bwd, rtol=rtol, atol=atol)
    # reverse the backward leg (excluding the duplicated IC) and prepend
    X = np.concatenate([X_bwd[:, :0:-1, :], X_fwd], axis=1)
    t = np.concatenate([-t_bwd[:0:-1], t_fwd])
    return TrajectoryData(
        X=X,
        t=t,
        meta={
            "system": system.name,
            "horizon": horizon,
            "backward": backward,
            "ts": ts,
        },
    )
