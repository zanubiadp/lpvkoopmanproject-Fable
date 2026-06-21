"""Near-limit-cycle error-budget diagnostic (oracle lift).

Question: in the paper configuration (5 s data, N = 20), is the large
near-limit-cycle prediction error caused by (a) interpolation of the
eigenfunctions, or (b) the fitted exponential model itself?

Method: replace the data-driven interpolated lift Phi(x0) by an "oracle"
lift that uses the known vector field: integrate backward from x0 to the
boundary circle r = 0.05 (giving the exact transit time tau and boundary
point), interpolate the learned boundary values g around the circle, and
evaluate phi(x0) = e^{lambda tau} g exactly as the construction defines it.
This removes all spatial-interpolation error. If predictions still fail
near the limit cycle, the deficiency is the truncated exponential model
evaluated at time depth tau (which exceeds the fitted window there), not
the interpolation.

Observed result (250 test ICs, 1 s horizon, errors in %):

                       interior (d>0.05)   near-LC band (d<0.05)
    interpolated lift        ~1.8                ~260
    oracle lift              ~1.8               ~1600

i.e. interpolation is NOT the bottleneck anywhere, and the near-LC failure
is intrinsic to the N=20 / 5 s model class: for x0 near the limit cycle the
construction needs e^{lambda tau} at tau ~ 5..40 s while the eigenvalues
were identified on t in [0, 5] - growing exponentials extrapolate away from
the bounded quasi-periodic truth. (Consistent with Korda & Mezic's own
caveat that performance degrades near the limit cycle, and with the
learnability obstructions discussed by Colbrook & Mezic: the spectrum on
the limit cycle, {i k omega_0}, is not contained in any finite truncation
of the interior lattice with positive real parts.)

Run:  python examples/diagnose_near_lc.py   (~2 min)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lpvkoopman import FitConfig, fit_koopman_model, vdp_scaled
from lpvkoopman.metrics import relative_rmse_percent
from lpvkoopman.vdp_benchmark import limit_cycle, make_training_data, sample_interior


def oracle_lift(model, system, x0, radius=0.05, t_max=40.0):
    """Exact flow-based lift: backward to the IC circle, then e^{lam tau} g."""
    hit = lambda t, x: x[0] ** 2 + x[1] ** 2 - radius**2
    hit.terminal = True
    hit.direction = -1
    sol = solve_ivp(
        lambda t, x: -system.f(x), (0.0, t_max), x0,
        rtol=1e-10, atol=1e-12, events=hit,
    )
    if sol.t_events[0].size == 0:
        return None
    tau = float(sol.t_events[0][0])
    p = sol.y_events[0][0]
    th = np.arctan2(p[1], p[0])

    # Boundary values around the circle: training ICs are equispaced in
    # angle starting at theta = 0 (circle_initial_conditions), so G columns
    # map directly to angles for periodic 1-D interpolation.
    z = []
    for comp in model.components:
        G = comp.G
        m_t = G.shape[1]
        th_b = 2 * np.pi * np.arange(m_t) / m_t  # circle ICs are equispaced
        order = np.argsort(th_b)
        col = 0
        for a_, b_ in comp.spec.pairs:
            a = np.interp(th, th_b[order], G[col, order], period=2 * np.pi)
            b = np.interp(th, th_b[order], G[col + 1, order], period=2 * np.pi)
            ea, cb, sb = np.exp(a_ * tau), np.cos(b_ * tau), np.sin(b_ * tau)
            z += [ea * (a * cb + b * sb), ea * (-a * sb + b * cb)]
            col += 2
        for r_ in comp.spec.reals:
            g = np.interp(th, th_b[order], G[col, order], period=2 * np.pi)
            z.append(np.exp(r_ * tau) * g)
            col += 1
    return np.array(z)


def main() -> int:
    system = vdp_scaled()
    data = make_training_data(n_traj=100, horizon=5.0)
    model, _ = fit_koopman_model(data, FitConfig(n_eig=20, optimize=True))

    lc = limit_cycle(system)
    rng = np.random.default_rng(7)
    x0s = sample_interior(lc, 250, rng)
    t_pred = np.arange(101) * 0.01
    x_true = system.simulate_many(x0s, t_pred)
    d_lc = cKDTree(lc).query(x0s)[0]

    t0 = time.perf_counter()
    e_interp, e_oracle, keep = [], [], []
    for j, x0 in enumerate(x0s):
        z_o = oracle_lift(model, system, x0)
        if z_o is None:
            continue
        keep.append(j)
        e_oracle.append(relative_rmse_percent(
            model.propagate(z_o, t_pred) @ model.C.T, x_true[j]))
        e_interp.append(relative_rmse_percent(
            model.predict(x0, t_pred), x_true[j]))
    e_interp, e_oracle = np.array(e_interp), np.array(e_oracle)
    d = d_lc[keep]
    print(f"evaluated {len(keep)} ICs in {time.perf_counter()-t0:.0f}s\n")
    print(f"{'lift':14s} {'interior mean':>14s} {'near-LC mean':>14s} {'overall median':>15s}")
    for nm, e in [("interpolated", e_interp), ("oracle (flow)", e_oracle)]:
        print(f"{nm:14s} {e[d >= 0.05].mean():13.2f}% {e[d < 0.05].mean():13.2f}% "
              f"{np.median(e):14.2f}%")
    print("\nConclusion: oracle == interpolated in the interior (model-limited),"
          "\noracle >= interpolated near the LC (extrapolation in time depth,"
          "\nnot spatial interpolation, is the binding constraint).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
