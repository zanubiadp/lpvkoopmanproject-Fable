"""Numerical cross-check of the lpvkoopman eigenvalue-learning core against
the thesis MATLAB implementation (run via GNU Octave).

Same data in -> compare:
  1. projected cost J and its gradient against
     getCostGradientKordacc_re_fast.m (real conjugate-pair formulation,
     thesis Appendix A; identical up to the reference's column scaling,
     which provably does not affect J, the gradient, or the fitted values);
  2. fitted trajectory values L*q (boundary least squares, paper eq. 35);
  3. J against eigOptim_grad.m, the discrete-time complex formulation used
     by build_A.m, evaluated at mu = exp(lambda * Ts) (the two
     parametrizations span the same space, so J must agree).

Usage:
    python crosscheck/run_crosscheck.py [--matlab-repo PATH] [--octave OCTAVE]

Exits nonzero if any check fails.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lpvkoopman import circle_initial_conditions, generate_data, vdp_scaled
from lpvkoopman.spectrum import Spectrum
from lpvkoopman.varpro import (
    basis_matrix,
    complex_basis_matrix,
    fit_boundary,
    projection_cost_grad,
)

HERE = Path(__file__).resolve().parent


def build_problem():
    """Small, well-conditioned VdP test problem (shared by both sides)."""
    sys_ = vdp_scaled()
    data = generate_data(
        sys_, circle_initial_conditions(0.1, 10), horizon=1.5, ts=0.01
    )
    spec = Spectrum(pairs=[[0.5, 1.5], [-0.4, 3.3]], reals=[0.3])
    H = data.component(0)  # x1 along trajectories, (Ms+1, Mt)
    return data, spec, H


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--matlab-repo",
        default=os.environ.get(
            "MATLAB_REPO", str(HERE.parents[2] / "Koopman-MPC-thesis-MATLAB")
        ),
        help="path to the Koopman-MPC-thesis-MATLAB repository",
    )
    ap.add_argument("--octave", default="octave-cli")
    args = ap.parse_args()

    data, spec, H = build_problem()
    theta = spec.pack()  # same layout as the MATLAB x: [Re1 Im1 ... | reals]
    h_stacked = H.flatten(order="F")  # trajectory-major stacking

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        np.savetxt(td / "in_theta.csv", theta, delimiter=",")
        np.savetxt(td / "in_h.csv", h_stacked, delimiter=",")
        np.savetxt(td / "in_t.csv", data.t, delimiter=",")
        np.savetxt(
            td / "in_meta.csv",
            [[spec.n_pairs, data.n_traj, data.ts]],
            delimiter=",",
        )
        env = dict(os.environ, MATLAB_REPO=str(Path(args.matlab_repo).resolve()))
        proc = subprocess.run(
            [args.octave, "--no-gui", "-q", str(HERE / "crosscheck.m")],
            cwd=td,
            env=env,
            capture_output=True,
            text=True,
        )
        sys.stdout.write(proc.stdout)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            print("FAIL: octave run failed")
            return 1

        J_ref = float(np.loadtxt(td / "out_J_real.csv"))
        grad_ref = np.loadtxt(td / "out_grad_real.csv")
        grad_fd_ref = np.loadtxt(td / "out_grad_fd.csv")
        hhat_ref = np.loadtxt(td / "out_hhat_real.csv")
        J_disc_ref = float(np.loadtxt(td / "out_J_discrete.csv"))

    # --- lpvkoopman side -------------------------------------------------
    J_py, grad_py = projection_cost_grad(theta, spec.n_pairs, data.t, H)
    V = basis_matrix(spec, data.t)
    G, _ = fit_boundary(V, H)
    hhat_py = (V @ G).flatten(order="F")

    mu = np.exp(data.ts * spec.to_complex())
    Vd = complex_basis_matrix(np.log(mu) / data.ts, data.t)  # == e^{lam t_k}
    Gd, *_ = np.linalg.lstsq(Vd, H.astype(complex), rcond=None)
    J_disc_py = float(np.sum(np.abs(H - Vd @ Gd) ** 2))

    def rel(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return float(
            np.max(np.abs(a - b)) / max(1e-300, float(np.max(np.abs(b))))
        )

    # Tolerance notes:
    #  * The reference J is computed via an explicit Gram-matrix inverse
    #    (inv(L'L)), which costs ~8 digits relative to a QR/SVD solve, so
    #    J agreement is checked at 1e-6 (the reference's own eigOptim_grad,
    #    matched at 1e-9 below, confirms the tighter value).
    #  * Gradients: the two independent analytic implementations must agree
    #    within the reference's conditioning noise (1e-3 headroom; observed
    #    ~2e-5), and both must be consistent with the finite-difference
    #    gradient of the reference cost within the FD scheme's own accuracy
    #    (its step is limited from below by the reference J's noise).
    checks = [
        ("cost J (real cc formulation)", rel(J_py, J_ref), 1e-6),
        ("py grad vs reference analytic grad", rel(grad_py, grad_ref), 1e-3),
        ("py grad vs FD of reference cost", rel(grad_py, grad_fd_ref), 5e-3),
        ("fitted values L q == V G", rel(hhat_py, hhat_ref), 1e-8),
        ("cost J (discrete complex, eigOptim_grad)", rel(J_disc_py, J_disc_ref), 1e-9),
    ]
    ok = True
    print(f"\n{'check':45s} {'rel. diff':>12s}  tol")
    for name, r, tol in checks:
        status = "OK " if r <= tol else "FAIL"
        ok &= r <= tol
        print(f"{name:45s} {r:12.3e}  {tol:.0e}  {status}")
    print(
        f"\nJ_py = {J_py:.12e}\nJ_oct= {J_ref:.12e}\n"
        f"J_disc_py = {J_disc_py:.12e}\nJ_disc_oct= {J_disc_ref:.12e}"
    )
    print("reference analytic grad vs its own FD grad: rel diff %.3e"
          % rel(grad_ref, grad_fd_ref))
    print("py analytic grad vs reference analytic grad: rel diff %.3e"
          % rel(grad_py, grad_ref))
    print("CROSSCHECK", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
