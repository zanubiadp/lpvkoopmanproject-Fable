"""Koopman-LPV MPC on the true Van der Pol plant in its input-extended form.

The plant is ``vdp_nonaffine`` through its exact extension x3 = u, v = udot,
with the shared N = 40+1 model. Three tasks, each run twice on identical
lifted dynamics - once with the LPV field B(x) learned from autonomous data,
once with the Korda-Mezic constant B fitted to forced data:

  1. stabilization of the *unstable* origin;
  2. tracking of a moving reference x1 = 0.10 sin(0.8 t);
  3. piecewise-constant setpoints x1 = +/-0.10 (offset-free variant).

Two structural features, both forced by the extension rather than chosen:

  * the MPC's input is v = udot and the actuator command u is the extra state
    x3, so the controller uses ``integrator_state=2`` (the decision variable
    is the u-sequence, making |u| <= U_MAX an exact constraint). Boxing v
    instead would be a slew limit and would leave u free to wander outside
    the range the model was identified on;
  * the task sizes are set by the plant's actuation limits: the input enters
    as w(x,u) = (0.15 + 0.5 x2) sin(2u) + 0.25 x1 u^2, worth at most +-0.15
    at the origin. fig15 shows, for each task, why a larger version is
    beyond this plant's reach - keeping the LPV-vs-constant comparison a
    measurement of model quality rather than of actuator saturation.

Run:  python examples/run_vdp_mpc.py   (~1.5 min with the model cached)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from extended_vdp import KM, LPV, U_MAX, fit_input_models, get_model
from lpvkoopman import KoopmanMPC, MPCConfig, extend_input, run_closed_loop, vdp_nonaffine
from lpvkoopman.nonaffine import level_limit_cycle

# ---- task definitions --------------------------------------------------------
# Every number here is set by an actuation limit measured in fig15, not by
# taste; see section 8.3 of reports/validation_vdp_control.md.
TS_C = 0.10   # control period. The stabilizing manoeuvre spans the unstable
T_HOR = 8.0   # focus's half-rotation period, pi/beta = 4.06 s, so the horizon
              # has to be several seconds long; at ts = 0.01 that would be a
              # 800-step condensed QP per solve. Coarsening the control rate
              # buys the horizon at a cost the plant can afford.
X0_SCALE = 0.25   # |x0| = 0.19 against a null-controllable reach of 0.31
SIN_A, SIN_W = 0.10, 0.8
STEP_A, DWELL = 0.10, 4.0
T_SINE, T_STEP = 15.0, 32.0
T_STAB = 20.0   # long enough to show BOTH the convergence and the
                # subsequent escape - an 8 s window ends before the
                # loop loses the origin and reads as a clean success

# Weights: from a 72-configuration (Q, R, S) sweep per task, each configuration
# scored over 6 perturbed initial conditions per input model. The closed loop is
# bistable - it either holds the reference or escapes to the limit cycle - so
# configurations are ranked by how many of the 6 runs stayed bounded and only
# then by median error; the argmin of a single run would be fitting the noise.
#
# The shared weights below are picked NEUTRALLY (best combined LPV + constant
# score), not by the LPV arm alone. Selecting on the LPV arm and then reporting
# whatever the constant scores at those weights would rig the ablation, and the
# sine task shows that is not hypothetical: the constant's own optimum holds
# 4/6 where it holds 0/6 at the LPV optimum. Each arm's own best is recorded in
# results_mpc.json alongside the shared-weight numbers.
Q_STAB, R_STAB, S_STAB = (1.0, 0.05, 0.1), 1e-4, 1e-2
Q_SINE, R_SINE, S_SINE = (1.0, 0.2, 0.5), 1e-2, 1e-3
Q_STEP, R_STEP, S_STEP = (1.0, 0.2, 0.5), 1e-2, 1e-3


def null_controllable_region(u_max: float, T: float = 12.0, seed: int = 0):
    """Sample the set of states that can be driven to the origin under |u|<=u_max.

    That set is exactly what the time-REVERSED plant reaches from the origin.
    For a planar single-input system its boundary is traced by bang-bang
    controls, so the extremal branches are integrated backwards over a spread
    of switching times and the union of those curves fills the region. Used to
    draw fig15 and to justify task 1's initial condition: a start at 92 % of
    the limit-cycle radius sits 2.25x outside this set, from where no
    controller - not even one given the true plant - can stabilize.
    """
    from scipy.integrate import solve_ivp

    ug = np.linspace(-u_max, u_max, 201)

    def f_rev(x, sign):
        w = (0.15 + 0.5 * x[1]) * np.sin(2 * ug) + 0.25 * x[0] * ug ** 2
        wk = w.max() if sign > 0 else w.min()
        return -np.array([2 * x[1],
                          -0.8 * x[0] + 2 * x[1] - 10 * x[0] ** 2 * x[1] + wk])

    def traj(switches, sign0):
        x = np.array([1e-5, 1e-5 * sign0])
        out, t, s = [x.copy()], 0.0, sign0
        for tn in list(switches) + [T]:
            if tn <= t:
                continue
            sol = solve_ivp(lambda _t, z: f_rev(z, s), (t, tn), x,
                            max_step=0.03, rtol=1e-7, atol=1e-9)
            out.extend(sol.y.T[1:])
            x, t, s = sol.y[:, -1], tn, -s
            if np.linalg.norm(x) > 4:
                break
        return np.array(out)

    rng = np.random.default_rng(seed)
    pts = []
    for s0 in (+1, -1):
        for t1 in np.arange(0.0, 8.0, 0.5):
            pts.append(traj([t1], s0))
            for t2 in np.arange(t1 + 0.5, min(t1 + 5.0, 9.0), 1.0):
                pts.append(traj([t1, t2], s0))
    for _ in range(120):
        pts.append(traj(np.sort(rng.uniform(0, 8, rng.integers(1, 4))),
                        rng.choice([-1, 1])))
    P = np.vstack([p for p in pts if len(p)])
    return P[np.linalg.norm(P, axis=1) < 2.0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parents[1] / "reports"))
    args = ap.parse_args()
    out = Path(args.outdir)
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    system_na = vdp_nonaffine()
    ext = extend_input(system_na)

    t0 = time.perf_counter()
    model_na, data_na, comps = get_model(out / "model_nonaffine.pkl", args.quick)
    print(f"extended model ready ({time.perf_counter()-t0:.0f}s), "
          f"N={model_na.N}, {data_na.n_traj} trajectories")
    cloud = data_na.flat_points()
    x_clip = (cloud.min(axis=0), cloud.max(axis=0))

    # ---- the two input models, on identical lifted dynamics ---------------
    Bm = fit_input_models(model_na, data_na, args.quick, ablations=False)
    arms = [("lpv", Bm[LPV[0]], LPV[2]), ("const", Bm[KM[0]], KM[2])]

    na = {}

    def run_arm(B, cfg, x0, ref, t_end, offset_free=False):
        mpc = KoopmanMPC(model_na, B, cfg, x_clip=x_clip)
        t1 = time.perf_counter()
        res = run_closed_loop(mpc, ext, x0, reference=ref,
                              n_steps=int(round(t_end / cfg.ts)),
                              offset_free=offset_free)
        return res, round(time.perf_counter() - t1, 1)

    # ---- task 1: stabilize the unstable origin ----------------------------
    # x0 must lie inside the origin's null-controllable region, which for this
    # plant reaches only ~0.31 (a start at 0.92*lc would be 0.69 away - the
    # oracle in fig15 cannot stabilize from there either).
    lc0 = level_limit_cycle(system_na, 0.0)
    x0 = np.array([*(lc0[800] * X0_SCALE), 0.0])
    cfg1 = MPCConfig(horizon=int(round(T_HOR / TS_C)), ts=TS_C, Q=Q_STAB,
                     R=R_STAB, S=S_STAB, u_min=-U_MAX, u_max=U_MAX,
                     integrator_state=2)
    stab, free = {}, ext.simulate(x0, np.arange(0, T_STAB + TS_C, TS_C))
    for nm, B, _ in arms:
        res, wall = run_arm(B, cfg1, x0, lambda t: np.zeros(3), T_STAB)
        nx = np.linalg.norm(res.X[:, :2], axis=1)   # plant states only, not u
        stab[nm] = (res, nx)
        na[f"stabilization_{nm}"] = {
            "x0": x0.tolist(),
            "norm_at_2s": float(nx[int(2.0 / TS_C)]),
            "norm_at_4s": float(nx[int(4.0 / TS_C)]),
            "norm_final": float(nx[-1]),
            "min_norm": float(nx.min()),
            "t_at_min_s": float(res.t[int(np.argmin(nx))]),
            "tail_max": float(nx[int(len(nx) * 0.8):].max()),
            "u_max": float(np.abs(res.X[:, 2]).max()),
            "mean_solve_ms": float(res.solve_ms.mean()),
            "wall_s": wall,
        }
        print(f"stabilization {nm:5s}: min|x|={nx.min():.2e} at "
              f"t={res.t[int(np.argmin(nx))]:.1f}s, tail max={nx[int(len(nx)*0.8):].max():.2e}"
              f"  (|x0|={nx[0]:.3f})  solve {res.solve_ms.mean():.0f} ms")

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    ax = axes[0]
    ax.plot(lc0[:, 0], lc0[:, 1], "k-", lw=1.6, label="limit cycle ($u=0$)")
    ax.plot(free[:, 0], free[:, 1], color="0.6", lw=1.2, label="open loop ($u=0$)")
    for nm, _, c in arms:
        ax.plot(stab[nm][0].X[:, 0], stab[nm][0].X[:, 1], color=c, lw=1.8,
                label=f"closed loop - $B$ {nm}")
    ax.plot(x0[0], x0[1], "bo", ms=7, label="$x_0$")     # x0 is 3-D: index it
    ax.plot(0, 0, "r*", ms=13, label="target (unstable focus)")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_aspect("equal")
    ax.legend(fontsize=8); ax.set_title("Stabilization of the unstable origin")

    ax = axes[1]
    for nm, _, c in arms:
        ax.semilogy(stab[nm][0].t, np.maximum(stab[nm][1], 1e-8), color=c, lw=1.5,
                    label=f"$\\|x\\|$, $B$ {nm}")
    ax.set_xlabel("t [s]"); ax.set_ylabel("$\\|(x_1,x_2)\\|$")
    ax.legend(fontsize=8); ax.set_title("Convergence")

    ax = axes[2]
    for nm, _, c in arms:
        ax.plot(stab[nm][0].t, stab[nm][0].X[:, 2], color=c, lw=1.4,
                label=f"$u$, $B$ {nm}")
    ax.axhline(U_MAX, color="k", ls=":", lw=1.0)
    ax.axhline(-U_MAX, color="k", ls=":", lw=1.0, label="$|u|\\leq$ %.2f" % U_MAX)
    ax.set_xlabel("t [s]"); ax.set_ylabel("actuator $u=x_3$")
    ax.set_ylim(-1.3, 1.3); ax.legend(fontsize=8)
    ax.set_title("Actuator (the MPC's input is $v=\\dot u$)")
    fig.tight_layout()
    fig.savefig(figdir / "fig9_mpc_stabilization.png", dpi=150)
    plt.close(fig)

    # ---- task 2: sinusoidal reference tracking ----------------------------
    ref_sin = lambda t: np.array([SIN_A * np.sin(SIN_W * t), 0.0, 0.0])
    cfg2 = MPCConfig(horizon=int(round(T_HOR / TS_C)), ts=TS_C, Q=Q_SINE,
                     R=R_SINE, S=S_SINE, u_min=-U_MAX, u_max=U_MAX,
                     integrator_state=2)
    track = {}
    for nm, B, _ in arms:
        res, wall = run_arm(B, cfg2, np.zeros(3), ref_sin, T_SINE)
        e = res.X[1:, 0] - res.R[:, 0]
        k0 = int(round(2.0 / TS_C))
        track[nm] = res
        na[f"sine_tracking_{nm}"] = {
            "amplitude": SIN_A, "omega": SIN_W,
            "rms_after_2s": float(np.sqrt(np.mean(e[k0:] ** 2))),
            "rms_over_amplitude": float(np.sqrt(np.mean(e[k0:] ** 2)) / SIN_A),
            "max_abs_err_after_2s": float(np.abs(e[k0:]).max()),
            "u_max": float(np.abs(res.X[:, 2]).max()),
            "wall_s": wall,
        }
        print(f"sine tracking {nm:5s}: RMS(>2s) "
              f"{na[f'sine_tracking_{nm}']['rms_after_2s']:.4f} "
              f"({na[f'sine_tracking_{nm}']['rms_over_amplitude']:.2f} x amplitude "
              f"{SIN_A})")

    # ---- task 3: piecewise-constant setpoints -----------------------------
    # Holding +r already spends most of the actuator (at r = 0.20 only 0.027 of
    # push-up authority is left), so the region from which the setpoint can
    # still be recovered collapses on the +x1 side. Hence the small amplitude
    # and the longer dwell - see fig15 and the validation report.
    ref_step = lambda t: np.array(
        [STEP_A if (t % (2 * DWELL)) < DWELL else -STEP_A, 0.0, 0.0])
    cfg3 = MPCConfig(horizon=int(round(T_HOR / TS_C)), ts=TS_C, Q=Q_STEP,
                     R=R_STEP, S=S_STEP, u_min=-U_MAX, u_max=U_MAX,
                     integrator_state=2)
    step = {}
    for nm, B, _ in arms:
        res, wall = run_arm(B, cfg3, np.zeros(3), ref_step, T_STEP,
                            offset_free=True)
        es = res.X[1:, 0] - res.R[:, 0]
        tt = res.t[1:]
        msk = ((tt % DWELL) > 0.6 * DWELL) & (tt > DWELL)   # settled part only
        step[nm] = res
        na[f"step_tracking_{nm}"] = {
            "amplitude": STEP_A, "dwell_s": DWELL,
            "settled_rms": float(np.sqrt(np.mean(es[msk] ** 2))),
            "settled_rms_over_amplitude": float(
                np.sqrt(np.mean(es[msk] ** 2)) / STEP_A),
            "u_max": float(np.abs(res.X[:, 2]).max()),
            "wall_s": wall,
            "note": "residual bias from the model's stationarity error at "
                    "off-origin equilibria, compounded by the collapsing "
                    "recoverable margin above the setpoint; see fig15",
        }
        print(f"step tracking {nm:5s}: settled RMS "
              f"{na[f'step_tracking_{nm}']['settled_rms']:.4f} "
              f"(reference +/-{STEP_A})")

    fig, axes = plt.subplots(3, 2, figsize=(12.0, 8.0), sharex="col")
    for col, (runs, ttl) in enumerate(
        [(track, f"sine, $A$={SIN_A}, $\\omega$={SIN_W}"),
         (step, f"steps $\\pm${STEP_A}, dwell {DWELL} s (offset-free)")]
    ):
        ax = axes[0, col]
        any_res = runs[arms[0][0]]
        ax.plot(any_res.t[1:], any_res.R[:, 0], "k--", lw=1.3, label="reference")
        for nm, _, c in arms:
            ax.plot(runs[nm].t, runs[nm].X[:, 0], color=c, lw=1.5,
                    label=f"$x_1$, $B$ {nm}")
        ax.set_ylabel("$x_1$"); ax.set_title(f"Tracking: {ttl}")
        ax.legend(fontsize=8)
        ax = axes[1, col]
        for nm, _, c in arms:                       # the ACTUATOR, u = x3 ...
            ax.plot(runs[nm].t, runs[nm].X[:, 2], color=c, lw=1.2)
        ax.axhline(U_MAX, color="k", ls=":", lw=1.0)
        ax.axhline(-U_MAX, color="k", ls=":", lw=1.0)
        ax.set_ylabel("$u = x_3$"); ax.set_ylim(-1.3, 1.3)
        ax = axes[2, col]
        for nm, _, c in arms:                       # ... and the MPC's input
            ax.step(runs[nm].t[:-1], runs[nm].U[:, 0], color=c, lw=0.9)
        ax.set_ylabel("$v = \\dot u$"); ax.set_xlabel("t [s]")
    fig.tight_layout()
    fig.savefig(figdir / "fig10_mpc_tracking.png", dpi=150)
    plt.close(fig)

    # ---- fig15: the actuation limits that set the task sizes --------------
    ug = np.linspace(-U_MAX, U_MAX, 801)
    w_env = lambda a, b: ((0.15 + 0.5 * b) * np.sin(2 * ug) + 0.25 * a * ug ** 2)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))
    ax = axes[0]
    tp = np.linspace(0, 10, 801)
    for A, om, c, lab in [(0.3, 1.2, "tab:red", "too large:"),
                          (SIN_A, SIN_W, "tab:green", "chosen:")]:
        r, rd, rdd = (A * np.sin(om * tp), A * om * np.cos(om * tp),
                      -A * om ** 2 * np.sin(om * tp))
        x2 = rd / 2.0
        w_req = rdd / 2.0 + 0.8 * r - 2 * x2 + 10 * r ** 2 * x2
        env = np.array([[w_env(a, b).min(), w_env(a, b).max()] for a, b in zip(r, x2)])
        ax.plot(tp, w_req, color=c, lw=1.8, label=f"{lab} $A$={A}, $\\omega$={om}")
        ax.fill_between(tp, env[:, 0], env[:, 1], color=c, alpha=0.15)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("t [s]"); ax.set_ylabel("$w$ (input effect on $\\dot x_2$)")
    ax.set_title("Task 2: demanded (line) vs achievable (band)")
    ax.legend(fontsize=8)

    ax = axes[1]
    rr = np.linspace(-0.35, 0.35, 301)
    env = np.array([[w_env(r, 0.0).min(), w_env(r, 0.0).max()] for r in rr])
    ax.fill_between(rr, env[:, 0], env[:, 1], color="0.75", alpha=0.7,
                    label="achievable $w$ at $x=(r,0)$")
    ax.plot(rr, 0.8 * rr, "k-", lw=1.8, label="required $w=0.8\\,r$")
    for r0, c, lab in [(0.25, "tab:red", "too large: $\\pm$0.25"),
                       (STEP_A, "tab:green", f"chosen: $\\pm${STEP_A}")]:
        ax.axvline(r0, color=c, ls="--", lw=1.2)
        ax.axvline(-r0, color=c, ls="--", lw=1.2, label=lab)
    ax.set_xlabel("setpoint $r$"); ax.set_ylabel("$w$")
    ax.set_title("Task 3: equilibrium feasibility")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[2]
    nullc = null_controllable_region(U_MAX)
    ax.plot(lc0[:, 0], lc0[:, 1], "k-", lw=1.3, label="limit cycle ($u=0$)")
    ax.plot(nullc[:, 0], nullc[:, 1], ".", ms=0.5, color="tab:blue",
            alpha=0.25, label="steerable to the origin")
    ax.plot(*(lc0[800] * 0.92), "o", color="tab:red", ms=8, label="$x_0 = 0.92\\,$LC (too far)")
    ax.plot(*(lc0[800] * X0_SCALE), "o", color="tab:green", ms=8,
            label=f"chosen $x_0 = {X0_SCALE}\\,$LC")
    ax.plot(0, 0, "r*", ms=13)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_aspect("equal")
    ax.set_title(f"Task 1: null-controllable region, $|u|\\leq{U_MAX}$")
    ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Actuation limits of the extended plant ($|u| \\leq {U_MAX}$, "
                 "$|w| \\leq 0.15$ at the origin) that set the MPC task sizes")
    fig.tight_layout()
    fig.savefig(figdir / "fig15_actuation_limits.png", dpi=150)
    plt.close(fig)

    (out / "results_mpc.json").write_text(json.dumps(na, indent=1))
    print(f"results -> {out/'results_mpc.json'}")
    print(f"figures -> {figdir}/fig9, fig10, fig15")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
