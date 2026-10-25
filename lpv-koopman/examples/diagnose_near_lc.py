"""Near-limit-cycle error-budget diagnostic (oracle lift) for the
input-extended Van der Pol model.

Question: in the paper configuration (5 s data, N = 20+1), is the large
near-limit-cycle prediction error caused by (a) interpolation of the
eigenfunctions over the 3-D cloud of u-planes, or (b) the fitted exponential
model itself?

Method: on a *trained* u-plane, replace the data-driven interpolated lift
Phi(x0, u) by an "oracle" lift that uses the known vector field: integrate
the frozen-u slice backward from x0 to that level's boundary circle
(r = 0.05 around the level's equilibrium), giving the exact transit time tau
and boundary point; interpolate that level's learned boundary values g around
the circle, and evaluate phi(x0) = e^{lambda tau} g exactly as the
construction defines it (the x3 block, lambda = 0, returns u). This removes
all spatial-interpolation error - in-plane and across planes. If predictions
still fail near the limit cycle, the deficiency is the truncated exponential
model evaluated at time depth tau (which exceeds the fitted window there),
not the interpolation.

Evaluated on the planes u = -0.6, 0 and +0.6 (u = 0 is the Korda & Mezic
system); results are printed per plane and pooled - see
reports/validation_vdp.md section 4.

Run:  python examples/diagnose_near_lc.py   (~2 min)
"""
from __future__ import annotations

import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

from extended_vdp import TS, fit_extended_model, make_data, truncate
from lpvkoopman import vdp_nonaffine
from lpvkoopman.metrics import relative_rmse_percent
from lpvkoopman.nonaffine import equilibrium, frozen, level_limit_cycle
from lpvkoopman.vdp_benchmark import sample_interior

PLANES = (-0.6, 0.0, 0.6)


def oracle_lift(model, sysna, x0, u, cols, radius=0.05, t_max=40.0):
    """Exact flow-based lift on the plane x3 = u: backward to that level's IC
    circle, then e^{lam tau} g with g interpolated over the level's columns."""
    xeq = equilibrium(sysna, u)
    hit = lambda t, x: (x[0] - xeq[0]) ** 2 + (x[1] - xeq[1]) ** 2 - radius**2
    hit.terminal = True
    hit.direction = -1
    sol = solve_ivp(
        lambda t, x: -sysna.f(x, u), (0.0, t_max), x0,
        rtol=1e-10, atol=1e-12, events=hit,
    )
    if sol.t_events[0].size == 0:
        return None
    tau = float(sol.t_events[0][0])
    p = sol.y_events[0][0] - xeq
    th = np.arctan2(p[1], p[0])

    # Boundary values around the circle: a level's training ICs are
    # equispaced in angle starting at theta = 0 (circle_initial_conditions),
    # so its G columns map directly to angles. With only 12 ICs per level
    # (30 deg apart) linear interpolation in angle would itself be a sizable
    # error source, so a periodic cubic spline is used.
    th_b = 2 * np.pi * np.arange(len(cols) + 1) / len(cols)
    th = np.mod(th, 2 * np.pi)
    z = []
    for comp in model.components:
        G = comp.G[:, cols]
        g = CubicSpline(th_b, np.column_stack([G, G[:, :1]]), axis=1,
                        bc_type="periodic")(th)
        col = 0
        for a_, b_ in comp.spec.pairs:
            a, b = g[col], g[col + 1]
            ea, cb, sb = np.exp(a_ * tau), np.cos(b_ * tau), np.sin(b_ * tau)
            z += [ea * (a * cb + b * sb), ea * (-a * sb + b * cb)]
            col += 2
        for r_ in comp.spec.reals:
            z.append(np.exp(r_ * tau) * g[col])
            col += 1
    return np.array(z), tau


def main() -> int:
    sysna = vdp_nonaffine()
    t0 = time.perf_counter()
    data = truncate(make_data(), 5.0)
    model, _ = fit_extended_model(data, n_half=10)
    print(f"paper-config extended model (N={model.N}) fitted in "
          f"{time.perf_counter()-t0:.0f}s")
    levels = np.asarray(data.meta["u_levels"])
    n_per = data.meta["n_traj_per_level"]

    t_pred = np.arange(101) * TS
    rng = np.random.default_rng(7)
    pooled = {"interpolated": [], "oracle (flow)": [], "d": [], "tau": []}
    print(f"\n{'plane':>6s} {'lift':14s} {'interior mean':>14s} {'near-LC mean':>14s} "
          f"{'overall median':>15s}")
    for u in PLANES:
        lvl = int(np.argmin(np.abs(levels - u)))
        cols = np.arange(lvl * n_per, (lvl + 1) * n_per)
        lc = level_limit_cycle(sysna, u)
        x0s = sample_interior(lc, 250, rng)
        x_true = frozen(sysna, u).simulate_many(x0s, t_pred)
        d_lc = cKDTree(lc).query(x0s)[0]
        e_interp, e_oracle, keep, taus = [], [], [], []
        for j, x0 in enumerate(x0s):
            res = oracle_lift(model, sysna, x0, u, cols)
            if res is None:
                continue
            z_o, tau = res
            keep.append(j)
            taus.append(tau)
            e_oracle.append(relative_rmse_percent(
                (model.propagate(z_o, t_pred) @ model.C.T)[:, :2], x_true[j]))
            e_interp.append(relative_rmse_percent(
                model.predict(np.array([*x0, u]), t_pred)[:, :2], x_true[j]))
        d = d_lc[keep]
        for nm, e in [("interpolated", np.array(e_interp)),
                      ("oracle (flow)", np.array(e_oracle))]:
            print(f"{u:+6.1f} {nm:14s} {e[d >= 0.05].mean():13.2f}% "
                  f"{e[d < 0.05].mean():13.2f}% {np.median(e):14.2f}%")
            pooled[nm].append(e)
        pooled["d"].append(d)
        pooled["tau"].append(np.array(taus))
    d = np.concatenate(pooled["d"])
    tau = np.concatenate(pooled["tau"])
    for nm in ("interpolated", "oracle (flow)"):
        e = np.concatenate(pooled[nm])
        print(f"{'all':>6s} {nm:14s} {e[d >= 0.05].mean():13.2f}% "
              f"{e[d < 0.05].mean():13.2f}% {np.median(e):14.2f}%")
    e_i = np.concatenate(pooled["interpolated"])
    e_o = np.concatenate(pooled["oracle (flow)"])
    print(f"\n({len(d)} ICs evaluated, {time.perf_counter()-t0:.0f}s total)")
    print("oracle / interpolated error by time depth tau (median ratio):")
    for lo_, hi_ in [(0, 3), (3, 5), (5, 7), (7, np.inf)]:
        m = (tau >= lo_) & (tau < hi_)
        if m.any():
            print(f"   tau in [{lo_}, {hi_}) s: {np.median(e_o[m] / e_i[m]):6.2f}"
                  f"  (n={m.sum()}, interpolated median {np.median(e_i[m]):.2f}%)")
    print("Within the 5 s fitted window the two lifts should agree if the model"
          "\nis the bottleneck; beyond it the oracle extrapolates e^{lambda tau}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
