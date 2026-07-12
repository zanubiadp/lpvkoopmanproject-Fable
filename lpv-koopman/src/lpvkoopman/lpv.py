"""LPV extension of the Koopman eigenfunction model.

For a control-affine system  xdot = f(x) + g_c(x) u  whose autonomous part
generated the eigenfunctions Phi, the lifted dynamics are exact (Iacob et
al. 2024):

    zdot = A z + B(x) u,      B(x) = (dPhi/dx)(x) g_c(x),

with the same block-diagonal A as the autonomous model. B is
state-dependent (LPV); this module provides

  * ``BMap``       - evaluates B(x) from the learned model's gradients
                     (MLS by default), plus a constant-B reduction for
                     LTI baselines;
  * ``discretize`` - exact zero-order-hold pair (Ad, Gamma) with
                     Bd(x) = Gamma B(x), computed blockwise in closed form
                     (no matrix exponentials of the full lift);
  * ``predict_forced`` - lifted LPV prediction under a known input signal,
                     evaluating B along the model's own predicted state.

No forced/input-perturbed experiments are used anywhere: B(x) comes
entirely from the autonomous model - the central claim of the thesis this
implements.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gradients import lift_jacobian
from .model import KoopmanEigenModel

Array = np.ndarray


@dataclass
class BMap:
    """State-dependent lifted input matrix B(x) = (dPhi/dx) g_c(x)."""

    model: KoopmanEigenModel
    g: callable  # control vector field g_c(x) -> (n,) or (n, m)
    method: str = "mls"
    options: dict | None = None

    def __call__(self, x: Array) -> Array:
        """B at a single state: (N, m)."""
        x = np.asarray(x, dtype=float).reshape(-1)
        J = lift_jacobian(self.model, x, method=self.method, **(self.options or {}))
        gx = np.asarray(self.g(x), dtype=float)
        if gx.ndim == 1:
            gx = gx[:, None]
        return J @ gx

    def batch(self, X: Array) -> Array:
        """B at many states: (Q, N, m)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        J = lift_jacobian(self.model, X, method=self.method, **(self.options or {}))
        G = np.stack([np.atleast_2d(np.asarray(self.g(x), dtype=float).T).T for x in X])
        return np.einsum("qij,qjk->qik", J, G)

    def constant(self, sample_points: Array, rng=None, max_samples: int = 2000) -> Array:
        """LTI baseline: average B over a sample of states (N, m)."""
        X = np.atleast_2d(sample_points)
        if len(X) > max_samples:
            rng = rng or np.random.default_rng(0)
            X = X[rng.choice(len(X), max_samples, replace=False)]
        return self.batch(X).mean(axis=0)


def _monomial_exponents(n_vars: int, degree: int) -> list[tuple[int, ...]]:
    """All exponent tuples with total degree <= degree, in the deterministic
    order (first variable's exponent outermost, ascending) that for n = 2
    reproduces the original 2-D column layout x1^i x2^j, i + j <= d."""
    if n_vars == 1:
        return [(i,) for i in range(degree + 1)]
    return [
        (i, *rest)
        for i in range(degree + 1)
        for rest in _monomial_exponents(n_vars - 1, degree - i)
    ]


def _poly_design(X: Array, degree: int) -> Array:
    X = np.atleast_2d(X)
    return np.column_stack(
        [
            np.prod([X[:, k] ** e for k, e in enumerate(exps)], axis=0)
            for exps in _monomial_exponents(X.shape[1], degree)
        ]
    )


@dataclass
class PolynomialBField:
    """Smooth global model of the LPV input matrix, B(x) ~ sum_k c_k beta_k(x).

    Raw pointwise gradient estimates of B(x) are exact in expectation but
    noisy, and near the limit cycle their magnitude explodes together with
    the eigenfunction gradients; rolling the lifted model forward with such
    samples is numerically unstable (each step injects O(||B_err||) into
    expanding modes). Fitting one low-order polynomial per lifted row over
    the operating region - weighting samples by eigenfunction-PDE quality -
    keeps every property we need (smooth, bounded, cheap, still built from
    autonomous data only) and makes forced rollouts stable. Degree 0
    recovers a constant-B LTI baseline.
    """

    coef: Array  # (n_terms, N, m)
    degree: int

    def __call__(self, x: Array) -> Array:
        P = _poly_design(np.atleast_2d(np.asarray(x, dtype=float)), self.degree)
        return np.einsum("t,tnm->nm", P[0], self.coef)

    @property
    def n_inputs(self) -> int:
        return self.coef.shape[2]


def fit_b_field(
    model: KoopmanEigenModel,
    g,
    sample_points: Array,
    degree: int = 4,
    f=None,
    method: str = "mls",
    options: dict | None = None,
) -> PolynomialBField:
    """Fit a PolynomialBField to MLS gradient samples of B(x).

    sample_points : (Q, n) states spread over the operating region.
    f : autonomous vector field; if given, samples are weighted by
        1 / (1 + (r / median r)^2) with r the eigenfunction-PDE residual,
        downweighting locations where the gradient estimate is unreliable.
    """
    from .gradients import pde_residual

    X = np.atleast_2d(np.asarray(sample_points, dtype=float))
    bmap = BMap(model, g, method=method, options=options)
    Bs = bmap.batch(X)  # (Q, N, m)
    if f is not None:
        r = pde_residual(model, f, X, method=method, **(options or {}))
        w = 1.0 / (1.0 + (r / max(np.median(r), 1e-12)) ** 2)
    else:
        w = np.ones(len(X))
    P = _poly_design(X, degree)
    sw = np.sqrt(w)[:, None]
    Q, N, m = Bs.shape
    coef, *_ = np.linalg.lstsq(sw * P, sw * Bs.reshape(Q, N * m), rcond=None)
    return PolynomialBField(coef=coef.reshape(-1, N, m), degree=degree)


@dataclass
class KnnBField:
    """Local (k-nearest-neighbor) smoother of raw B(x) samples.

    The polynomial field imposes one global shape on B(x); when the true
    field has localized structure (input-extended systems: the eigenfunction
    x3-derivatives inherit every kink of the input nonlinearity) a global
    low-degree fit flattens the real variation while still letting outlier
    samples bend it far from the data. Averaging the k nearest samples with
    combined distance x reliability weights keeps the local structure,
    suppresses isolated bad samples, and is still built from autonomous
    data only.
    """

    points: Array  # (Q, n) sample locations
    Bs: Array  # (Q, N, m) raw B samples
    w: Array  # (Q,) reliability weights (e.g. from the PDE residual)
    k: int = 25
    blend: float = 0.0
    """Confidence blending toward the global weighted-mean B. The gradient
    samples are spatially heterogeneous: excellent in the interior, garbage
    near the limit cycle where the eigenfunction gradients explode - and
    there a local average of bad samples is still bad, however the
    neighbors are weighted against each other. With blend = c > 0 the
    output is  alpha B_local + (1 - alpha) B_global  with
    alpha = w_local / (w_local + c w_global): regions whose neighborhoods
    are as reliable as the global average keep their local structure,
    low-confidence regions degrade gracefully to the (robust) constant."""

    def __post_init__(self):
        from scipy.spatial import cKDTree

        self._tree = cKDTree(self.points)
        wsum = self.w.sum()
        self._B_global = np.einsum("q,qnm->nm", self.w / wsum, self.Bs)
        self._w_mean = float(self.w.mean())

    def __call__(self, x: Array) -> Array:
        x = np.asarray(x, dtype=float).reshape(-1)
        d, idx = self._tree.query(x, k=min(self.k, len(self.points)))
        scale = max(d.max(), 1e-12)
        ww = self.w[idx] * ((1.0 - np.minimum(d / scale, 1.0) ** 3) ** 3 + 1e-3)
        B_local = np.einsum("q,qnm->nm", ww / ww.sum(), self.Bs[idx])
        if self.blend <= 0:
            return B_local
        w_loc = float(self.w[idx].mean())
        alpha = w_loc / (w_loc + self.blend * self._w_mean)
        return alpha * B_local + (1.0 - alpha) * self._B_global

    @property
    def n_inputs(self) -> int:
        return self.Bs.shape[2]


def fit_b_knn(
    model: KoopmanEigenModel,
    g,
    sample_points: Array,
    k: int = 25,
    f=None,
    method: str = "mls",
    options: dict | None = None,
    blend: float = 0.0,
) -> KnnBField:
    """Sample B(x) at ``sample_points`` and wrap the samples in a
    ``KnnBField`` (same sampling and PDE-residual weighting as
    ``fit_b_field``, local averaging instead of a global polynomial)."""
    from .gradients import pde_residual

    X = np.atleast_2d(np.asarray(sample_points, dtype=float))
    Bs = BMap(model, g, method=method, options=options).batch(X)
    if f is not None:
        r = pde_residual(model, f, X, method=method, **(options or {}))
        w = 1.0 / (1.0 + (r / max(np.median(r), 1e-12)) ** 2)
    else:
        w = np.ones(len(X))
    return KnnBField(points=X, Bs=Bs, w=w, k=k, blend=blend)


def fit_constant_b_forced(
    model: KoopmanEigenModel,
    trajectories: list[tuple[Array, Array]],
    ts: float,
) -> Array:
    """Korda & Mezic-style constant lifted input matrix from *forced* data.

    Their control pipeline keeps A (the eigenvalues) from the autonomous
    construction and identifies one constant B by linear least squares on
    input-perturbed experiments. With the exact ZOH pair (Ad, Gamma) and
    z_k = Phi(x_k) lifted from the measured states, each step gives one
    linear equation in B:

        z_{k+1} - Ad z_k = Gamma B u_k,

    stacked over all steps of all experiments and solved in one lstsq
    (Gamma is invertible blockwise, so B = Gamma^{-1} M with M the
    regressed map). This is the LTI baseline the LPV B(x) is compared
    against - note it consumes forced experiments, which the LPV
    construction does not need.

    trajectories : list of (X, U) with X (K+1, n) measured states and
    U (K,) or (K, m) the ZOH inputs applied. Returns B of shape (N, m).

    Steps whose lifted endpoints fall outside the training cloud's convex
    hull are dropped: the nearest-neighbor lift fallback there produces
    discontinuous z-jumps that read as enormous fake input responses and
    wreck the regression (measured: ||B|| inflated by ~5 orders of
    magnitude without the filter).
    """
    Ad, Gam = discretize(model, ts)
    rows_r, rows_u = [], []
    for X, U in trajectories:
        X = np.asarray(X, dtype=float)
        Z = model.lift(X)  # (K+1, N)
        inside = ~np.isnan(model._interp(X)).any(axis=1)
        U = np.atleast_2d(np.asarray(U, dtype=float).T).T  # (K, m)
        if len(U) != len(Z) - 1:
            raise ValueError("need exactly one input per step")
        keep = inside[1:] & inside[:-1]
        rows_r.append((Z[1:] - Z[:-1] @ Ad.T)[keep])
        rows_u.append(U[keep])
    R = np.vstack(rows_r)  # (K_tot, N)
    U = np.vstack(rows_u)  # (K_tot, m)
    if len(U) == 0:
        raise ValueError("no regression steps left inside the training hull")
    M, *_ = np.linalg.lstsq(U, R, rcond=None)  # R ~ U (Gamma B)^T
    return np.linalg.solve(Gam, M.T)


def discretize(model: KoopmanEigenModel, ts: float) -> tuple[Array, Array]:
    """Exact ZOH discretization of the lifted pair (A, I): returns (Ad, Gamma)
    with z+ = Ad z + Gamma B u for B frozen over the step.

    Ad = exp(A ts);  Gamma = int_0^ts exp(A s) ds, both blockwise:
      pair block  A_b = alpha I + beta J (J = [[0,1],[-1,0]], J^2 = -I):
        exp(A_b t) = e^{alpha t} (cos(beta t) I + sin(beta t) J)
        Gamma_b    = A_b^{-1} (exp(A_b ts) - I),
                     A_b^{-1} = (alpha I - beta J)/(alpha^2 + beta^2)
      real block: e^{alpha ts} and expm1(alpha ts)/alpha (-> ts as alpha -> 0).
    """
    N = model.N
    Ad = np.zeros((N, N))
    Gam = np.zeros((N, N))
    eye2 = np.eye(2)
    Jrot = np.array([[0.0, 1.0], [-1.0, 0.0]])
    col = 0
    for comp in model.components:
        for a, b in comp.spec.pairs:
            E = np.exp(a * ts) * (np.cos(b * ts) * eye2 + np.sin(b * ts) * Jrot)
            denom = a * a + b * b
            if denom > 1e-16:
                Ainv = (a * eye2 - b * Jrot) / denom
                G = Ainv @ (E - eye2)
            else:
                G = ts * eye2
            Ad[col : col + 2, col : col + 2] = E
            Gam[col : col + 2, col : col + 2] = G
            col += 2
        for r in comp.spec.reals:
            Ad[col, col] = np.exp(r * ts)
            Gam[col, col] = np.expm1(r * ts) / r if abs(r) > 1e-12 else ts
            col += 1
    return Ad, Gam


def predict_forced(
    model: KoopmanEigenModel,
    bmap: BMap | Array,
    x0: Array,
    u_seq: Array,
    ts: float,
    x_clip: tuple[Array, Array] | None = None,
) -> Array:
    """Predict the forced response from x0 under the ZOH input sequence.

    u_seq : (K,) or (K, m) - input held constant on each interval [k ts,
    (k+1) ts). Returns predicted states (K+1, n). B is re-evaluated each
    step at the model's own predicted state (pure model rollout - the true
    trajectory is never consulted). Pass a constant matrix instead of a
    ``BMap`` for an LTI baseline.

    x_clip : optional (lo, hi) box, typically the training-data bounds. The
    state fed to a state-dependent ``bmap`` is clipped into the box first:
    B(x) is only identified over the sampled region, and a polynomial field
    extrapolates violently outside it (a diverging rollout then overflows
    within a few steps instead of just being wrong). Standard LPV
    scheduling-variable saturation; ignored for constant B.
    """
    u_seq = np.atleast_2d(np.asarray(u_seq, dtype=float).T).T  # (K, m)
    Ad, Gam = discretize(model, ts)
    z = model.lift(np.asarray(x0, dtype=float))
    X = np.empty((len(u_seq) + 1, model.C.shape[0]))
    X[0] = model.C @ z
    for k, uk in enumerate(u_seq):
        if callable(bmap):
            xb = X[k] if x_clip is None else np.clip(X[k], x_clip[0], x_clip[1])
            B = bmap(xb)
        else:
            B = bmap
        z = Ad @ z + Gam @ (B @ uk)
        X[k + 1] = model.C @ z
    return X
