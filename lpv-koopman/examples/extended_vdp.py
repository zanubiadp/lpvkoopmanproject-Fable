"""Shared protocol for every experiment in examples/: the non-affine Van der
Pol plant (``vdp_nonaffine``) in its exact input-extended form.

    x1dot = 2 x2
    x2dot = -0.8 x1 + 2 x2 - 10 x1^2 x2 + w(x1, x2, x3)
    x3dot = v                      (x3 := u, new input v := udot)

The extended system is input-affine with the constant control field
g_ext = (0, 0, 1); all of the input nonlinearity w sits in the autonomous
part and is learned by the eigenfunctions Phi(x1, x2, x3). The input-free
plane x3 = 0 is exactly the scaled Van der Pol of Korda & Mezic (2020)
(w(x, 0) = 0), which is what makes their Table I a valid reference for the
autonomous benchmark.

This module owns everything the scripts must agree on, so that every
figure is produced by the same model and the same two input matrices:

  * ``fit_extended_model`` / ``get_model`` - the cached 3-D eigenfunction
    model (x3 fitted exactly as a lambda = 0 component);
  * ``fit_input_models`` - the input matrices compared throughout:
      lpv_knn          LPV B(x) = dPhi/dx3 from autonomous data (ours)
      km_const_forced  Korda-Mezic constant B regressed on forced data
    plus the ablations reported in tables only (lpv_poly3, const_auto).
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lpvkoopman import (
    FitConfig,
    extend_input,
    fit_b_field,
    fit_b_knn,
    fit_constant_b_forced,
    fit_koopman_model,
    make_extended_training_data,
    simulate_true,
    vdp_nonaffine,
)
from lpvkoopman.nonaffine import level_limit_cycle
from lpvkoopman.spectrum import dmd_eigenvalues
from lpvkoopman.systems import TrajectoryData
from lpvkoopman.vdp_benchmark import sample_interior

TS = 0.01
U_MAX = 1.15  # test signals stay just inside the trained range [-1.2, 1.2]
U_TRAIN = 1.2

# the two models every comparison figure shows: (results key, label, color)
LPV = ("lpv_knn", "LPV $B(x)$ (ours, autonomous data)", "tab:red")
KM = ("km_const_forced", "constant $B$ (Korda-Mezic, forced data)", "tab:green")


def make_data(quick: bool = False, horizon: float = 7.0) -> TrajectoryData:
    """Autonomous extended-system experiments at constant-u levels."""
    n_levels, n_traj = (17, 10) if quick else (25, 12)
    return make_extended_training_data(
        vdp_nonaffine(),
        np.linspace(-U_TRAIN, U_TRAIN, n_levels),
        n_traj_per_level=n_traj,
        horizon=horizon,
        ts=TS,
    )


def truncate(data: TrajectoryData, horizon: float) -> TrajectoryData:
    """The first ``horizon`` seconds of every trajectory (e.g. 5 s data from
    the 7 s experiments - same initial conditions, shorter records)."""
    k = int(round(horizon / data.ts)) + 1
    return TrajectoryData(X=data.X[:, :k], t=data.t[:k],
                          meta={**data.meta, "horizon": horizon})


def fit_extended_model(data: TrajectoryData, n_half: int = 20, optimize: bool = True):
    """Fit the extended model with ``n_half`` eigenfunctions per physical
    state (x1, x2) and the exact lambda = 0 component for x3: N = 2 n_half + 1."""
    base = dmd_eigenvalues(data.X, data.ts)
    base = base[np.abs(base) > 1e-6]  # drop the exact lambda=0 of x3
    return fit_koopman_model(
        data,
        FitConfig(
            n_eig=[n_half, n_half, 1],
            optimize=optimize,
            constant_components=(2,),
            alpha_bounds=(-20.0, 1.5),
            base_eigs=base,
            boundary_rcond=1e-3,
        ),
    )


def get_model(cache: Path, quick: bool = False):
    """The reference model (7 s data, N = 41), cached: (model, data, components)."""
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    data = make_data(quick)
    model, report = fit_extended_model(data)
    cache.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump((model, data, report.components), open(cache, "wb"))
    return model, data, report.components


def b_sample_points(quick: bool = False) -> np.ndarray:
    """Where B(x) is sampled: interiors of the frozen-u limit cycles at levels
    between (and on) the training planes."""
    sysna = vdp_nonaffine()
    rng = np.random.default_rng(5)
    n_levels = 16 if quick else 24
    n_pts = 80 if quick else 150
    pts = []
    for u in np.linspace(-U_MAX, U_MAX, n_levels):
        xy = sample_interior(level_limit_cycle(sysna, u), n_pts, rng, shrink=0.9)
        pts.append(np.column_stack([xy, np.full(len(xy), u)]))
    return np.vstack(pts)


def fd_options(data: TrajectoryData) -> dict:
    """Per-dimension FD steps: fine in-plane, half the plane spacing across."""
    levels = np.asarray(data.meta["u_levels"])
    return dict(h=np.array([2e-3, 2e-3, (levels[1] - levels[0]) / 2]), order=2)


def forced_experiments(n: int = 60, K: int = 150, seed: int = 9):
    """Forced data for the Korda-Mezic regression: white-noise ZOH v from
    interior initial conditions, v zeroed whenever u would leave |u| <= U_MAX.
    (White noise decorrelates the input from the autonomous model residual -
    block inputs gave a visibly worse-biased regression.)"""
    sysna = vdp_nonaffine()
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        u0 = rng.uniform(-1.0, 1.0)
        x0 = sample_interior(level_limit_cycle(sysna, u0), 1, rng, shrink=0.8)[0]
        v = rng.uniform(-3.0, 3.0, K)
        u = u0
        for k in range(K):
            un = u + v[k] * TS
            if abs(un) > U_MAX:
                v[k] = 0.0
            else:
                u = un
        out.append((simulate_true(sysna, x0, u0, v, TS), v))
    return out


def fit_input_models(model, data: TrajectoryData, quick: bool = False,
                     ablations: bool = True) -> dict:
    """All input matrices on the model's lifted dynamics, keyed as in the
    results files. Only ``lpv_knn`` and ``km_const_forced`` appear in figures."""
    ext = extend_input(vdp_nonaffine())
    pts = b_sample_points(quick)
    opts = fd_options(data)
    out = {
        "lpv_knn": fit_b_knn(model, ext.g, pts, k=20, f=ext.f, method="fd",
                             options=opts, blend=0.3),
    }
    if ablations:
        out["lpv_poly3"] = fit_b_field(model, ext.g, pts, degree=3, f=ext.f,
                                       method="fd", options=opts)
        out["const_auto"] = fit_b_field(model, ext.g, pts, degree=0, f=ext.f,
                                        method="fd", options=opts)
    out["km_const_forced"] = fit_constant_b_forced(model, forced_experiments(), TS)
    return out
