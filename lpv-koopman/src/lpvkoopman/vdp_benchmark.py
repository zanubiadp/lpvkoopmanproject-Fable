"""Van der Pol benchmark utilities replicating Korda & Mezic (2020), Sec. VI-A.

Protocol (uncontrolled rows of Table I):
  * training: 100 trajectories, 5 s, Ts = 0.01, initial conditions on a
    circle of radius 0.05 around the origin;
  * evaluation: prediction over a 1 s horizon from 500 initial conditions
    sampled uniformly over the interior of the limit cycle, error metric
    eq. (55): 100 * ||x_pred - x_true||_F / ||x_true||_F, averaged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path as MplPath

from .metrics import relative_rmse_percent
from .model import KoopmanEigenModel
from .systems import ContinuousSystem, TrajectoryData, circle_initial_conditions, generate_data, vdp_scaled

Array = np.ndarray


def make_training_data(
    n_traj: int = 100,
    horizon: float = 5.0,
    ts: float = 0.01,
    radius: float = 0.05,
    system: ContinuousSystem | None = None,
) -> TrajectoryData:
    system = system or vdp_scaled()
    return generate_data(
        system, circle_initial_conditions(radius, n_traj), horizon=horizon, ts=ts
    )


def limit_cycle(system: ContinuousSystem | None = None, dt: float = 1e-3) -> Array:
    """One period of the attracting limit cycle, as a closed (K, 2) polyline."""
    system = system or vdp_scaled()
    x = system.simulate(np.array([0.3, 0.0]), np.arange(0.0, 60.0, dt))
    tail = x[-int(20.0 / dt) :]
    th = np.arctan2(tail[:, 1], tail[:, 0])
    cross = np.where((th[:-1] < 0) & (th[1:] >= 0))[0]
    if len(cross) < 2:
        raise RuntimeError("failed to detect limit cycle period")
    return tail[cross[0] : cross[1] + 1]


def sample_interior(
    lc: Array,
    count: int,
    rng: np.random.Generator,
    shrink: float = 1.0,
) -> Array:
    """Uniform samples over the interior of the (optionally shrunk) limit cycle.

    ``shrink`` < 1 scales the boundary polygon toward the origin, which is
    useful for sampling "well inside" or, with a pair of calls, near-boundary
    bands.
    """
    poly = MplPath(lc * shrink)
    lo = (lc * shrink).min(axis=0)
    hi = (lc * shrink).max(axis=0)
    out = np.empty((count, 2))
    k = 0
    while k < count:
        cand = lo + (hi - lo) * rng.random((4 * (count - k), 2))
        keep = cand[poly.contains_points(cand)]
        take = min(len(keep), count - k)
        out[k : k + take] = keep[:take]
        k += take
    return out


@dataclass
class EvalResult:
    errors: Array  # (Q,) per-IC relative RMSE [%]
    x0s: Array  # (Q, 2)
    horizon: float
    lift_fallbacks: int  # how many test ICs fell outside the training hull

    @property
    def mean(self) -> float:
        return float(np.mean(self.errors))

    @property
    def median(self) -> float:
        return float(np.median(self.errors))

    def summary(self) -> str:
        e = self.errors
        return (
            f"mean {self.mean:.2f}%  median {self.median:.2f}%  "
            f"p90 {np.percentile(e, 90):.2f}%  max {e.max():.2f}%  "
            f"(n={len(e)}, hull fallbacks={self.lift_fallbacks})"
        )


def evaluate_model(
    model: KoopmanEigenModel,
    x0s: Array,
    horizon: float = 1.0,
    ts: float = 0.01,
    system: ContinuousSystem | None = None,
    x_true: Array | None = None,
) -> EvalResult:
    """Paper-protocol evaluation: eq. (55) error per IC over ``horizon``."""
    system = system or vdp_scaled()
    t = np.arange(int(round(horizon / ts)) + 1) * ts
    if x_true is None:
        x_true = system.simulate_many(x0s, t)
    z0 = model.lift(x0s)  # (Q, N) batch lift
    fallbacks = model.last_lift_fallbacks
    errors = np.empty(len(x0s))
    for j in range(len(x0s)):
        x_pred = model.propagate(z0[j], t) @ model.C.T
        errors[j] = relative_rmse_percent(x_pred, x_true[j])
    return EvalResult(
        errors=errors, x0s=x0s, horizon=horizon, lift_fallbacks=fallbacks
    )
