"""Koopman-LPV model predictive control on the lifted linear model.

Controller structure (the standard qLPV-Koopman MPC, as in the thesis):
at every control step

  1. lift the measured state, z = Phi(x);
  2. evaluate the LPV input matrix at the current state and freeze it over
     the horizon: Bd = Gamma B(x) (exact ZOH discretization from lpv.py);
  3. solve the condensed finite-horizon tracking problem

        min_U  sum_k ||y_k - r_k||_Q^2 + ||u_k||_R^2 + ||u_k - u_{k-1}||_S^2
        s.t.   z_{k+1} = Ad z_k + Bd u_k,  y_k = C z_k,
               u_min <= u_k <= u_max

     which, having only box constraints, is solved exactly as a bounded
     least-squares problem (scipy.optimize.lsq_linear - no extra QP
     dependency); apply u_0 and repeat.

The plant in the closed-loop simulator is the true nonlinear system,
integrated with solve_ivp under zero-order-hold inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import lsq_linear

from .lpv import BMap, discretize
from .model import KoopmanEigenModel
from .systems import ContinuousSystem

Array = np.ndarray


@dataclass
class MPCConfig:
    horizon: int = 80  # prediction/control horizon (steps)
    ts: float = 0.01
    Q: tuple = (1.0, 1.0)  # output weights (diagonal)
    R: float = 1e-3  # input magnitude weight
    S: float = 1e-2  # input increment weight
    u_min: float = -1.0
    u_max: float = 1.0


@dataclass
class MPCSolution:
    u0: Array
    U: Array  # (Np, m) planned inputs
    Y: Array  # (Np, n) predicted outputs
    cost: float


class KoopmanMPC:
    def __init__(
        self,
        model: KoopmanEigenModel,
        bmap: BMap | Array,
        config: MPCConfig | None = None,
    ):
        self.model = model
        self.bmap = bmap
        self.cfg = config or MPCConfig()
        self.Ad, self.Gam = discretize(model, self.cfg.ts)
        self.C = model.C
        self.n_out = self.C.shape[0]
        # C Ad^k for k = 1..Np (free response map), shared by every solve
        Np = self.cfg.horizon
        self._CAk = np.empty((Np, self.n_out, model.N))
        M = self.C.copy()
        for k in range(Np):
            M = M @ self.Ad
            self._CAk[k] = M

    def _input_matrix(self, x: Array) -> Array:
        B = self.bmap(x) if callable(self.bmap) else self.bmap
        return self.Gam @ B  # (N, m)

    def solve(self, x: Array, r_seq: Array, u_prev: Array | None = None) -> MPCSolution:
        """One MPC solve at measured state ``x`` for reference ``r_seq``.

        r_seq : (Np, n_out) reference over the horizon (broadcastable).
        """
        cfg = self.cfg
        Np = cfg.horizon
        z0 = self.model.lift(np.asarray(x, dtype=float))
        Bd = self._input_matrix(x)
        m = Bd.shape[1]

        r_seq = np.broadcast_to(
            np.asarray(r_seq, dtype=float), (Np, self.n_out)
        )
        u_prev = np.zeros(m) if u_prev is None else np.asarray(u_prev, float).reshape(m)

        # impulse-response blocks T_i = C Ad^i Bd, i = 0..Np-1
        T = np.empty((Np, self.n_out, m))
        v = Bd
        T[0] = self.C @ v
        for i in range(1, Np):
            v = self.Ad @ v
            T[i] = self.C @ v

        # free response and block-Toeplitz forced-response map
        F0 = self._CAk @ z0  # (Np, n_out)
        G = np.zeros((Np * self.n_out, Np * m))
        for k in range(Np):
            for j in range(k + 1):
                G[k * self.n_out : (k + 1) * self.n_out, j * m : (j + 1) * m] = T[k - j]

        sqQ = np.sqrt(np.asarray(cfg.Q, dtype=float))
        Wq = np.tile(sqQ, Np)[:, None]
        rows = [Wq * G]
        rhs = [(Wq[:, 0] * (r_seq - F0).reshape(-1))]
        if cfg.R > 0:
            rows.append(np.sqrt(cfg.R) * np.eye(Np * m))
            rhs.append(np.zeros(Np * m))
        if cfg.S > 0:
            D = np.eye(Np * m) - np.eye(Np * m, k=-m)
            d0 = np.zeros(Np * m)
            d0[:m] = u_prev
            rows.append(np.sqrt(cfg.S) * D)
            rhs.append(np.sqrt(cfg.S) * d0)
        Fls = np.vstack(rows)
        bls = np.concatenate(rhs)

        res = lsq_linear(
            Fls, bls, bounds=(cfg.u_min, cfg.u_max), method="trf",
            tol=1e-10, max_iter=200,
        )
        U = res.x.reshape(Np, m)
        Y = F0 + (G @ res.x).reshape(Np, self.n_out)
        return MPCSolution(u0=U[0], U=U, Y=Y, cost=float(res.cost))


@dataclass
class ClosedLoopResult:
    t: Array  # (K+1,)
    X: Array  # (K+1, n) true plant states
    U: Array  # (K, m) applied inputs
    R: Array  # (K, n_out) reference at each step
    solve_ms: Array  # (K,) per-step solve time


def run_closed_loop(
    mpc: KoopmanMPC,
    system: ContinuousSystem,
    x0: Array,
    reference,
    n_steps: int,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    offset_free: bool = False,
    disturbance_gain: float = 0.3,
) -> ClosedLoopResult:
    """Simulate the true nonlinear plant under the MPC in closed loop.

    ``reference``: callable t -> (n_out,) desired output, sampled over each
    horizon at the control rate. Plant: xdot = f(x) + g(x) u with ZOH u.

    ``offset_free`` enables a standard output-disturbance correction: the
    one-step output prediction error is low-pass filtered (gain
    ``disturbance_gain``) into an estimate d_hat, and the reference handed
    to the MPC is shifted by it. This removes part of the steady bias from
    plant/model mismatch at off-origin setpoints (see the validation
    report; the residual bias is a documented limitation).
    """
    import time as _time

    cfg = mpc.cfg
    ts = cfg.ts
    x = np.asarray(x0, dtype=float).copy()
    X = [x.copy()]
    U, Rlog, ms = [], [], []
    u_prev = None
    d_hat = np.zeros(mpc.n_out)
    for k in range(n_steps):
        tk = k * ts
        r_seq = np.stack([np.asarray(reference(tk + (i + 1) * ts), dtype=float)
                          for i in range(cfg.horizon)])
        if offset_free:
            r_seq = r_seq - d_hat
        t0 = _time.perf_counter()
        sol = mpc.solve(x, r_seq, u_prev=u_prev)
        ms.append(1e3 * (_time.perf_counter() - t0))
        u = sol.u0
        if offset_free:
            z = mpc.model.lift(x)
            x_pred = mpc.C @ (mpc.Ad @ z + mpc._input_matrix(x) @ u)
        forced = ContinuousSystem(
            f=lambda xx, _u=u: system.f(xx)
            + (np.asarray(system.g(xx), dtype=float).reshape(len(xx), -1) @ _u),
            n=system.n,
        )
        x = forced.simulate(x, np.array([0.0, ts]), rtol=rtol, atol=atol)[-1]
        if offset_free:
            d_hat = (1 - disturbance_gain) * d_hat + disturbance_gain * (x - x_pred)
        X.append(x.copy())
        U.append(u.copy())
        Rlog.append(np.asarray(reference(tk + ts), dtype=float))
        u_prev = u
    return ClosedLoopResult(
        t=np.arange(n_steps + 1) * ts,
        X=np.array(X),
        U=np.array(U),
        R=np.array(Rlog),
        solve_ms=np.array(ms),
    )
