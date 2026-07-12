"""Non-input-affine systems and the input-extension ("velocity form") trick.

Both Korda & Mezic (2020) and Iacob et al. (2024) note that a general
non-affine system

    x1dot = f1(x1, x2, u)
    x2dot = f2(x1, x2, u)

can be rewritten exactly as the *input-affine* extended system

    x1dot = f1(x1, x2, x3)
    x2dot = f2(x1, x2, x3)          with  x3 := u,  new input  v := udot,
    x3dot = v

so every method built for control-affine dynamics (eigenfunction learning
from autonomous data, the LPV input matrix B(x) = dPhi/dx g_c(x), constant-B
lifted predictors) applies unchanged - the thesis uses exactly this trick
for the (naturally non-affine) vehicle model. This module provides

  * ``NonAffineSystem``      - a 2-state plant with scalar non-affine input;
  * ``vdp_nonaffine``        - the benchmark plant: scaled Van der Pol with a
                               deliberately nasty input law (sign-flipping,
                               state-multiplied input gain);
  * ``extend_input``         - the exact affine extension as a
                               ``ContinuousSystem`` (g_ext = e_3 constant!);
  * data-collection helpers  - autonomous extended experiments: initial
                               conditions around the frozen-u equilibrium at
                               several constant u levels (thesis protocol);
  * ``simulate_true``        - ground-truth forced simulation (v applied ZOH,
                               hence u piecewise linear), used as the
                               reference in every comparison.

The key structural payoff: in the extended coordinates the control vector
field is the *constant* g_ext = (0, 0, 1), so B(x) = dPhi/dx3 - all of the
input nonlinearity w(x, u) moves into the autonomous part and is absorbed
by the eigenfunctions Phi(x1, x2, x3). A constant lifted B cannot represent
an input gain whose sign depends on (x2, u); the LPV B(x) can.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

from .systems import ContinuousSystem, TrajectoryData, circle_initial_conditions

Array = np.ndarray


@dataclass
class NonAffineSystem:
    """Two-state plant  xdot = f(x, u)  with scalar, non-affine input.

    Attributes
    ----------
    f:
        Right-hand side f(x, u); x of shape (2,) or (2, Q), u scalar or (Q,).
    dfdu:
        Analytic input gain df/du (same broadcasting); ground truth for the
        LPV-vs-constant-B gain comparison, never used in learning.
    u_range:
        Operating interval of u covered by the benchmark protocol.
    """

    f: Callable[[Array, Array], Array]
    dfdu: Callable[[Array, Array], Array]
    n: int = 2
    m: int = 1
    u_range: tuple[float, float] = (-1.0, 1.0)
    name: str = "nonaffine"


def vdp_nonaffine() -> NonAffineSystem:
    """Scaled Van der Pol with a strongly non-affine, state-coupled input.

        x1dot = 2 x2
        x2dot = -0.8 x1 + 2 x2 - 10 x1^2 x2 + w(x, u)
        w(x, u) = (0.15 + 0.5 x2) sin(2u) + 0.25 x1 u^2

    Designed so the linear(-in-u) and true behaviors differ as much as
    possible over u in [-1.2, 1.2] while every frozen-u slice keeps the
    Van der Pol structure (unstable focus + attracting limit cycle, verified
    numerically over the whole u range - so autonomous experiments still
    explore the state space at every level):

      * dw/du = 2 (0.15 + 0.5 x2) cos(2u) + 0.5 x1 u  flips sign in u at
        |u| = pi/4 ~ 0.785 (the sin(2u) crest) - beyond it, pushing u up
        pushes the state *down*;
      * the gain is multiplied by (0.15 + 0.5 x2), which itself changes sign
        at x2 = -0.3 - the same input action reverses direction depending on
        where the state is;
      * the 0.25 x1 u^2 term is curvature-only (zero linear part at u = 0):
        invisible to any model linearized around u = 0, and it couples the
        input to the state through x1.

    No constant lifted input matrix can reproduce a sign-indefinite,
    state-dependent gain; this is the stress test the LPV matrix is for.
    """

    def f(x: Array, u) -> Array:
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)
        x1, x2 = x[0], x[1]
        w = (0.15 + 0.5 * x2) * np.sin(2.0 * u) + 0.25 * x1 * u**2
        return np.stack(
            [2.0 * x2, -0.8 * x1 + 2.0 * x2 - 10.0 * (x1**2) * x2 + w]
        )

    def dfdu(x: Array, u) -> Array:
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)
        x1, x2 = x[0], x[1]
        dw = 2.0 * (0.15 + 0.5 * x2) * np.cos(2.0 * u) + 0.5 * x1 * u
        return np.stack([np.zeros_like(dw), dw])

    return NonAffineSystem(
        f=f, dfdu=dfdu, u_range=(-1.2, 1.2), name="vdp_nonaffine"
    )


# --------------------------------------------------------------------------
# exact input-affine extension
# --------------------------------------------------------------------------

def extend_input(sys: NonAffineSystem) -> ContinuousSystem:
    """The exact affine extension: state (x1, x2, x3 = u), input v = udot.

    Autonomous part: (f(x, x3), 0) - x3 is constant along autonomous flows.
    Control vector field: the constant g_ext = (0, 0, 1) - the extended
    system is control-affine by construction, whatever f looks like in u.
    """

    def f_ext(xe: Array) -> Array:
        xe = np.asarray(xe, dtype=float)
        dx = sys.f(xe[: sys.n], xe[sys.n])
        return np.concatenate([dx, np.zeros_like(xe[sys.n : sys.n + 1])])

    def g_ext(xe: Array) -> Array:
        xe = np.asarray(xe, dtype=float)
        shape = (sys.n + 1,) if xe.ndim == 1 else (sys.n + 1, xe.shape[1])
        out = np.zeros(shape)
        out[sys.n] = 1.0
        return out

    return ContinuousSystem(
        f=f_ext, n=sys.n + 1, name=f"{sys.name}_extended", g=g_ext
    )


def frozen(sys: NonAffineSystem, u: float) -> ContinuousSystem:
    """The 2-state autonomous slice at constant input u."""
    return ContinuousSystem(
        f=lambda x: sys.f(x, u), n=sys.n, name=f"{sys.name}_u{u:+.2f}"
    )


def equilibrium(sys: NonAffineSystem, u: float, x_guess: Array | None = None) -> Array:
    """Equilibrium of the frozen-u slice (the point the training circles
    surround - the analogue of the origin in the u = 0 benchmark)."""
    x0 = np.zeros(sys.n) if x_guess is None else np.asarray(x_guess, dtype=float)
    sol = fsolve(lambda x: sys.f(x, u), x0, xtol=1e-13)
    # fsolve's status flag is unreliable at tight xtol; accept on residual.
    if np.linalg.norm(sys.f(sol, u)) > 1e-9:
        raise RuntimeError(f"equilibrium search failed at u={u}")
    return sol


def level_limit_cycle(sys: NonAffineSystem, u: float, dt: float = 1e-3) -> Array:
    """One period of the frozen-u slice's limit cycle as a closed polyline.

    Same crossing detection as ``vdp_benchmark.limit_cycle`` but centered at
    the (u-dependent) equilibrium, since the whole cycle shifts with u.
    """
    xeq = equilibrium(sys, u)
    sysu = frozen(sys, u)
    x = sysu.simulate(xeq + np.array([0.3, 0.0]), np.arange(0.0, 60.0, dt))
    tail = x[-int(20.0 / dt):]
    rel = tail - xeq
    th = np.arctan2(rel[:, 1], rel[:, 0])
    cross = np.where((th[:-1] < 0) & (th[1:] >= 0))[0]
    if len(cross) < 2:
        raise RuntimeError(f"failed to detect limit cycle at u={u}")
    return tail[cross[0] : cross[1] + 1]


# --------------------------------------------------------------------------
# autonomous data collection on the extended system (thesis protocol)
# --------------------------------------------------------------------------

def make_extended_training_data(
    sys: NonAffineSystem,
    u_levels: Array,
    n_traj_per_level: int = 12,
    horizon: float = 6.0,
    ts: float = 0.01,
    radius: float = 0.05,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> TrajectoryData:
    """Autonomous experiments of the extended system: for each constant
    u level, initial conditions on a small circle around that level's
    equilibrium, integrated with u frozen (x3dot = 0).

    This is exactly the thesis protocol ("experiments from different initial
    conditions at different values of u_constant"): the extended system's
    autonomous flow lives on the planes x3 = const, so sweeping the levels is
    the only way to make the data cloud three-dimensional - which the
    eigenfunction interpolation and the dPhi/dx3 gradient both require.
    """
    u_levels = np.asarray(u_levels, dtype=float).reshape(-1)
    ext = extend_input(sys)
    n_steps = int(round(horizon / ts))
    t = np.arange(n_steps + 1) * ts
    X = np.empty((len(u_levels) * n_traj_per_level, n_steps + 1, sys.n + 1))
    j = 0
    for u in u_levels:
        xeq = equilibrium(sys, u)
        ics = circle_initial_conditions(radius, n_traj_per_level, center=tuple(xeq))
        for x0 in ics:
            xe0 = np.array([*x0, u])
            X[j] = ext.simulate(xe0, t, rtol=rtol, atol=atol)
            j += 1
    return TrajectoryData(
        X=X,
        t=t,
        meta={
            "system": ext.name,
            "u_levels": u_levels.tolist(),
            "n_traj_per_level": n_traj_per_level,
            "horizon": horizon,
            "ts": ts,
            "radius": radius,
        },
    )


# --------------------------------------------------------------------------
# ground-truth forced simulation
# --------------------------------------------------------------------------

def simulate_true(
    sys: NonAffineSystem,
    x0: Array,
    u0: float,
    v_seq: Array,
    ts: float,
    rtol: float = 1e-9,
    atol: float = 1e-11,
) -> Array:
    """Simulate the *true* non-affine plant under the extended-system input
    convention: v is zero-order-hold on each step, hence u(t) is piecewise
    linear from u0. Returns the extended trajectory (K+1, 3) including the
    u trace in the last column.

    Integrating the extended ODE with the exact plant f is byte-identical to
    integrating the original 2-state plant under the same piecewise-linear
    u(t) (the extension is exact, not an approximation) - asserted in tests.
    """
    ext = extend_input(sys)
    e_u = np.zeros(sys.n + 1)
    e_u[sys.n] = 1.0
    xe = np.array([*np.asarray(x0, dtype=float), float(u0)])
    out = np.empty((len(v_seq) + 1, sys.n + 1))
    out[0] = xe
    for k, vk in enumerate(np.asarray(v_seq, dtype=float)):
        sol = solve_ivp(
            lambda _t, z: ext.f(z) + e_u * vk,
            (0.0, ts),
            xe,
            rtol=rtol,
            atol=atol,
        )
        if not sol.success:
            raise RuntimeError(f"forced integration failed at step {k}")
        xe = sol.y[:, -1]
        out[k + 1] = xe
    return out


def u_profile_to_v(u_way: Array, ts: float) -> tuple[float, Array]:
    """Convert a desired piecewise-linear u profile (sampled every ts) into
    the extended-system input: initial u0 and the ZOH rate sequence v.

    u_way : (K+1,) waypoints; returns (u0, v of shape (K,)) with
    v_k = (u_{k+1} - u_k)/ts, so the reconstructed u(t) interpolates the
    waypoints exactly.
    """
    u_way = np.asarray(u_way, dtype=float).reshape(-1)
    return float(u_way[0]), np.diff(u_way) / ts
