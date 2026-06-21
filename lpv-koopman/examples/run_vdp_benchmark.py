"""Full Van der Pol benchmark: reproduces every number and figure in
reports/validation_vdp.md.

Protocol (Korda & Mezic 2020, Sec. VI-A, uncontrolled):
  * paper configuration: 100 trajectories x 5 s, Ts = 0.01, ICs on a circle
    of radius 0.05; N in {4, 8, 12, 16, 20}, lattice vs optimized;
  * extended configuration: same circle, 7 s trajectories; N in {20, 30, 40};
  * evaluation: 500 ICs uniform over the limit-cycle interior (seed 7),
    1 s horizon, error metric eq. (55), reported overall and split into the
    near-limit-cycle band (distance < 0.05) and the rest.

Run:  python examples/run_vdp_benchmark.py [--quick] [--outdir reports]
Runtime: ~3 minutes full, ~40 s with --quick.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lpvkoopman import FitConfig, fit_koopman_model, vdp_scaled
from lpvkoopman.spectrum import lattice
from lpvkoopman.vdp_benchmark import (
    evaluate_model,
    limit_cycle,
    make_training_data,
    sample_interior,
)

# Published reference values: Korda & Mezic (2020), Table I, uncontrolled,
# average over 500 random ICs in the interior of the limit cycle.
PAPER_TABLE1 = {
    "lattice": {4: 100.4, 8: 95.6, 12: 51.34, 16: 13.31, 20: 5.44},
    "optimized": {4: 24.0, 8: 9.7, 12: 5.1, 16: 2.4, 20: 1.4},
}
LC_OMEGA = 1.1074  # rad/s, fundamental frequency of the scaled-VdP limit cycle


def fit_and_eval(data, cfg, x0s, x_true, system, d_lc):
    t0 = time.perf_counter()
    model, rep = fit_koopman_model(data, cfg)
    fit_s = time.perf_counter() - t0
    ev = evaluate_model(model, x0s, system=system, x_true=x_true)
    e = ev.errors
    row = {
        "mean": ev.mean,
        "median": ev.median,
        "p90": float(np.percentile(e, 90)),
        "max": float(e.max()),
        "near_lc_mean": float(e[d_lc < 0.05].mean()),
        "interior_mean": float(e[d_lc >= 0.05].mean()),
        "hull_fallbacks": ev.lift_fallbacks,
        "fit_seconds": round(fit_s, 1),
        "spectra": [c["spectrum"] for c in rep.components],
        "dmd_base": [str(np.round(b, 4)) for b in rep.dmd_base],
    }
    return model, rep, ev, row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="reduced sweep for CI")
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parents[1] / "reports"))
    args = ap.parse_args()

    out = Path(args.outdir)
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    system = vdp_scaled()
    rng_test = np.random.default_rng(7)
    n_test = 150 if args.quick else 500

    print("== data generation ==")
    t0 = time.perf_counter()
    data5 = make_training_data(n_traj=100, horizon=5.0)
    data7 = make_training_data(n_traj=100, horizon=7.0)
    lc = limit_cycle(system)
    x0s = sample_interior(lc, n_test, rng_test)
    t_pred = np.arange(101) * 0.01
    x_true = system.simulate_many(x0s, t_pred)
    d_lc = cKDTree(lc).query(x0s)[0]
    print(f"   {time.perf_counter()-t0:.1f}s "
          f"(train 5s/7s, LC, {n_test} test ICs + truth)")

    results: dict = {"n_test": n_test, "paper_table1": PAPER_TABLE1}
    models = {}

    print("== paper configuration: 5 s data, Table I sweep ==")
    n_sweep = [12, 20] if args.quick else [4, 8, 12, 16, 20]
    results["paper_config"] = {}
    for n_eig in n_sweep:
        for opt in (False, True):
            tag = "optimized" if opt else "lattice"
            cfg = FitConfig(n_eig=n_eig, optimize=opt)
            model, rep, ev, row = fit_and_eval(
                data5, cfg, x0s, x_true, system, d_lc
            )
            results["paper_config"][f"N{n_eig}_{tag}"] = row
            paper = PAPER_TABLE1[tag].get(n_eig)
            print(f"   N={n_eig:<2d} {tag:9s} mean {row['mean']:7.2f}%  "
                  f"interior {row['interior_mean']:6.2f}%  near-LC "
                  f"{row['near_lc_mean']:8.2f}%  (paper full: {paper}%)")
            if n_eig == 20:
                models[f"paper_N20_{tag}"] = (model, ev)

    print("== extended configuration: 7 s data ==")
    results["extended_config"] = {}
    ext_sweep = [40] if args.quick else [20, 30, 40]
    for n_eig in ext_sweep:
        for opt in (False, True):
            tag = "optimized" if opt else "lattice"
            cfg = FitConfig(
                n_eig=n_eig, optimize=opt, alpha_bounds=(-20.0, 1.0)
            )
            model, rep, ev, row = fit_and_eval(
                data7, cfg, x0s, x_true, system, d_lc
            )
            results["extended_config"][f"N{n_eig}_{tag}"] = row
            print(f"   N={n_eig:<2d} {tag:9s} mean {row['mean']:7.2f}%  "
                  f"interior {row['interior_mean']:6.2f}%  near-LC "
                  f"{row['near_lc_mean']:8.2f}%  [{row['fit_seconds']}s]")
            if n_eig == max(ext_sweep):
                models[f"best_N{n_eig}_{tag}"] = (model, ev)

    best_key = f"best_N{max(ext_sweep)}_optimized"
    best_model, best_ev = models[best_key]

    # ------------------------------------------------------------------ figures
    print("== figures ==")
    train_pts = data7.flat_points()

    # 1. phase portrait: training cloud, LC, test ICs
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(train_pts[::23, 0], train_pts[::23, 1], ".", ms=1, color="0.75",
            label="training samples (7 s)")
    ax.plot(lc[:, 0], lc[:, 1], "k-", lw=2, label="limit cycle")
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(0.05 * np.cos(th), 0.05 * np.sin(th), "b-", lw=1.5,
            label="IC circle r=0.05")
    sc = ax.scatter(x0s[:, 0], x0s[:, 1], c=np.log10(best_ev.errors), s=14,
                    cmap="viridis", zorder=3)
    fig.colorbar(sc, ax=ax, label=r"$\log_{10}$ prediction error [%]")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title(f"Training data and test errors ({best_key})")
    ax.legend(loc="upper left", fontsize=8); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(figdir / "fig1_phase_portrait.png", dpi=150)
    plt.close(fig)

    # 2. eigenvalues: lattice vs optimized, with LC harmonics
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, key, ttl in [
        (axes[0], "paper_N20_optimized", "paper config (5 s, N=20)"),
        (axes[1], best_key, f"extended config (7 s, N={max(ext_sweep)})"),
    ]:
        mdl, _ = models[key]
        lat_base = lattice(np.array([1 + 0.7745966692j, 1 - 0.7745966692j]), 3)
        for k in range(-12, 13):
            ax.axhline(k * LC_OMEGA, color="0.9", lw=0.7, zorder=0)
        ax.scatter(lat_base.real, lat_base.imag, marker="s", s=45,
                   facecolor="none", edgecolor="tab:blue",
                   label="linearization lattice (deg<=3)")
        ev_ = mdl.eigenvalues()
        ax.scatter(ev_.real, ev_.imag, marker="x", s=45, color="tab:red",
                   label="learned")
        ax.set_xlabel(r"Re $\lambda$"); ax.set_title(ttl)
        ax.axvline(0, color="0.8", lw=0.7)
    axes[0].set_ylabel(r"Im $\lambda$")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Continuous-time eigenvalues "
                 "(horizontal lines: limit-cycle harmonics $k\\,\\omega_0$)")
    fig.tight_layout(); fig.savefig(figdir / "fig2_eigenvalues.png", dpi=150)
    plt.close(fig)

    # 3. error vs N against the published Table I
    if not args.quick:
        fig, ax = plt.subplots(figsize=(7.5, 5))
        Ns = n_sweep
        for tag, color in [("lattice", "tab:blue"), ("optimized", "tab:red")]:
            mine_int = [results["paper_config"][f"N{n}_{tag}"]["interior_mean"] for n in Ns]
            mine_full = [results["paper_config"][f"N{n}_{tag}"]["mean"] for n in Ns]
            paper = [PAPER_TABLE1[tag][n] for n in Ns]
            ax.semilogy(Ns, paper, "o--", color=color, alpha=0.45,
                        label=f"paper Table I ({tag})")
            ax.semilogy(Ns, mine_int, "s-", color=color,
                        label=f"ours, interior d>0.05 ({tag})")
            ax.semilogy(Ns, mine_full, "^:", color=color, alpha=0.8,
                        label=f"ours, full interior ({tag})")
        ext = [results["extended_config"][f"N{n}_optimized"]["mean"] for n in ext_sweep]
        ax.semilogy(ext_sweep, ext, "d-", color="tab:green", lw=2,
                    label="ours, 7 s data, full interior (optimized)")
        ax.set_xlabel("number of eigenfunctions N")
        ax.set_ylabel("mean prediction error over 1 s [%]")
        ax.set_title("Prediction error vs. lift dimension (uncontrolled VdP)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(figdir / "fig3_error_vs_N.png", dpi=150)
        plt.close(fig)

    # 4. example predictions with the best model
    show_idx = [int(np.argmin(np.abs(d_lc - q))) for q in
                (np.percentile(d_lc, [85, 50, 15, 2]))]
    fig, axes = plt.subplots(2, len(show_idx), figsize=(3.1 * len(show_idx), 5.4),
                             sharex=True)
    for col, j in enumerate(show_idx):
        xp = best_model.predict(x0s[j], t_pred)
        for rowi in range(2):
            ax = axes[rowi, col]
            ax.plot(t_pred, x_true[j][:, rowi], "b-", lw=1.8, label="true")
            ax.plot(t_pred, xp[:, rowi], "r--", lw=1.6, label="predicted")
            if rowi == 0:
                ax.set_title(f"d(x0, LC)={d_lc[j]:.3f}\nerr={best_ev.errors[j]:.2f}%",
                             fontsize=9)
            ax.set_ylabel(f"$x_{rowi+1}$" if col == 0 else "")
            if rowi == 1:
                ax.set_xlabel("t [s]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"True vs. predicted trajectories ({best_key})")
    fig.tight_layout(); fig.savefig(figdir / "fig4_predictions.png", dpi=150)
    plt.close(fig)

    # 5. error over time percentiles (best vs paper-config optimized)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for key, color in [("paper_N20_optimized", "tab:orange"), (best_key, "tab:green")]:
        mdl, _ = models[key]
        z0 = mdl.lift(x0s)
        errt = np.empty((len(x0s), len(t_pred)))
        for j in range(len(x0s)):
            xp = mdl.propagate(z0[j], t_pred) @ mdl.C.T
            errt[j] = np.linalg.norm(xp - x_true[j], axis=1)
        med = np.percentile(errt, 50, axis=0)
        lo, hi = np.percentile(errt, 10, axis=0), np.percentile(errt, 90, axis=0)
        ax.semilogy(t_pred, med, color=color, label=f"{key} (median)")
        ax.fill_between(t_pred, lo, hi, color=color, alpha=0.18)
    ax.set_xlabel("prediction time [s]")
    ax.set_ylabel(r"$\|x_{pred}-x_{true}\|$")
    ax.set_title("State error over the prediction horizon (median, 10-90%)")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(figdir / "fig5_error_vs_time.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ outputs
    np.savez(
        out / "benchmark_arrays.npz",
        x0s=x0s,
        d_lc=d_lc,
        errors_best=best_ev.errors,
        errors_paper_n20=models["paper_N20_optimized"][1].errors,
        limit_cycle=lc,
    )
    (out / "results.json").write_text(json.dumps(results, indent=1))
    print(f"\nresults  -> {out/'results.json'}")
    print(f"figures  -> {figdir}/fig*.png")
    b = results["extended_config"][f"N{max(ext_sweep)}_optimized"]
    print(f"\nHEADLINE ({best_key}): mean {b['mean']:.2f}%  median "
          f"{b['median']:.2f}%  near-LC {b['near_lc_mean']:.2f}%  "
          f"(paper N=20: 1.4%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
