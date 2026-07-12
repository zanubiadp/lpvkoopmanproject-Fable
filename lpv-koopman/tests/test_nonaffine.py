"""Tests for the non-affine benchmark: input extension, constant components,
n-D generalizations, and the Korda-Mezic constant-B regression."""
import numpy as np
import pytest

from lpvkoopman import (
    FitConfig,
    circle_initial_conditions,
    extend_input,
    fit_constant_b_forced,
    fit_koopman_model,
    generate_data,
    linear_system,
    make_extended_training_data,
    predict_forced,
    simulate_true,
    u_profile_to_v,
    vdp_nonaffine,
)
from lpvkoopman.gradients import lift_jacobian
from lpvkoopman.lpv import _monomial_exponents, _poly_design, discretize
from lpvkoopman.nonaffine import equilibrium, frozen
from lpvkoopman.systems import ContinuousSystem


# --------------------------------------------------------------------------
# the extension trick itself
# --------------------------------------------------------------------------

def test_extension_is_exact():
    """Simulating the extended affine system under ZOH v must reproduce the
    original non-affine plant under the corresponding piecewise-linear u -
    the extension is an exact rewrite, not an approximation."""
    sysna = vdp_nonaffine()
    ts, K = 0.01, 60
    rng = np.random.default_rng(0)
    x0 = np.array([0.2, -0.1])
    u_way = 0.8 * np.sin(np.linspace(0.0, 3.0, K + 1))
    u0, v = u_profile_to_v(u_way, ts)

    # reference: integrate the ORIGINAL 2-state plant with u(t) piecewise
    # linear (independent implementation of the same input signal)
    from scipy.integrate import solve_ivp

    x = x0.copy()
    ref = [x.copy()]
    for k in range(K):
        def rhs(t, xx, k=k):
            u_t = u_way[k] + (u_way[k + 1] - u_way[k]) * (t / ts)
            return sysna.f(xx, u_t)

        sol = solve_ivp(rhs, (0.0, ts), x, rtol=1e-10, atol=1e-12)
        x = sol.y[:, -1]
        ref.append(x.copy())
    ref = np.array(ref)

    xt = simulate_true(sysna, x0, u0, v, ts, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(xt[:, :2], ref, atol=1e-7)
    # and the u trace interpolates the waypoints exactly
    np.testing.assert_allclose(xt[:, 2], u_way, atol=1e-9)


def test_extended_system_is_input_affine():
    """g_ext is the constant e_3 regardless of state - the whole point."""
    sysna = vdp_nonaffine()
    ext = extend_input(sysna)
    for xe in [np.zeros(3), np.array([0.3, -0.5, 1.0])]:
        np.testing.assert_allclose(ext.g(xe), [0.0, 0.0, 1.0])
    # autonomous part freezes u: x3dot = 0
    xe = np.array([0.1, 0.2, 0.7])
    dx = ext.f(xe)
    assert dx[2] == 0.0
    np.testing.assert_allclose(dx[:2], sysna.f(xe[:2], 0.7))


def test_input_gain_flips_sign():
    """The designed plant is genuinely non-affine: the input gain df2/du
    changes sign both across u and across the state."""
    sysna = vdp_nonaffine()
    g_low = sysna.dfdu(np.zeros(2), 0.0)[1]
    g_high = sysna.dfdu(np.zeros(2), 1.2)[1]
    assert g_low > 0 > g_high  # flip in u (past the sin(2u) crest)
    g_xa = sysna.dfdu(np.array([0.0, 0.4]), 0.0)[1]
    g_xb = sysna.dfdu(np.array([0.0, -0.6]), 0.0)[1]
    assert g_xa > 0 > g_xb  # flip in x2 (through the 0.15 + 0.5 x2 factor)


def test_equilibria_track_u():
    sysna = vdp_nonaffine()
    for u in [-1.2, -0.4, 0.0, 0.6, 1.2]:
        xeq = equilibrium(sysna, u)
        np.testing.assert_allclose(sysna.f(xeq, u), 0.0, atol=1e-9)
        assert xeq[1] == pytest.approx(0.0, abs=1e-9)  # x1dot = 2 x2 = 0


# --------------------------------------------------------------------------
# pipeline pieces
# --------------------------------------------------------------------------

def test_extended_training_data_layout():
    sysna = vdp_nonaffine()
    levels = np.array([-0.5, 0.0, 0.5])
    data = make_extended_training_data(
        sysna, levels, n_traj_per_level=4, horizon=0.5, ts=0.05
    )
    assert data.X.shape == (12, 11, 3)
    # x3 constant along each trajectory, equal to its level
    np.testing.assert_allclose(data.X[:, :, 2].std(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(data.X[::4, 0, 2], levels)


def test_constant_component_fitted_exactly():
    """x3 = u gets the single eigenvalue 0 and boundary value = the level."""
    sysna = vdp_nonaffine()
    data = make_extended_training_data(
        sysna, np.array([-0.6, 0.2, 0.8]), n_traj_per_level=6, horizon=1.0, ts=0.02
    )
    model, report = fit_koopman_model(
        data,
        FitConfig(n_eig=[6, 6, 1], constant_components=(2,), optimize=False),
    )
    comp = model.components[2]
    assert comp.spec.n_pairs == 0 and comp.spec.n_reals == 1
    assert comp.spec.reals[0] == 0.0
    np.testing.assert_allclose(comp.G[0], data.X[:, 0, 2])
    assert report.components[2]["constant_component"] is True
    # prediction of x3 is exact from any training point
    t = np.arange(5) * 0.02
    xp = model.predict(data.X[7, 0], t)
    np.testing.assert_allclose(xp[:, 2], data.X[7, 0, 2], atol=1e-10)


def test_poly_design_nd_matches_2d_layout():
    """The generalized monomial design must reproduce the original 2-D
    column layout exactly (pickled B-fields stay valid)."""
    X = np.random.default_rng(1).normal(size=(7, 2))
    d = 3
    ref = np.column_stack(
        [X[:, 0] ** i * X[:, 1] ** j for i in range(d + 1) for j in range(d + 1 - i)]
    )
    np.testing.assert_allclose(_poly_design(X, d), ref)
    # 3-D: right count (total degree <= d in 3 vars) and correct values
    X3 = np.random.default_rng(2).normal(size=(5, 3))
    P = _poly_design(X3, 2)
    exps = _monomial_exponents(3, 2)
    assert P.shape == (5, 10) and len(exps) == 10
    k = exps.index((1, 0, 1))
    np.testing.assert_allclose(P[:, k], X3[:, 0] * X3[:, 2])


def test_mls_jacobian_3d_linear_lift():
    """n-D MLS: on a 3-D linear system the lift is linear, so the MLS
    Jacobian must recover the (constant) lift matrix."""
    A = np.array([[0.0, 1.0, 0.0], [-1.0, -0.2, 0.3], [0.1, 0.0, -0.5]])
    sys3 = linear_system(A)
    rng = np.random.default_rng(3)
    x0s = 0.2 * rng.normal(size=(30, 3))
    data = generate_data(sys3, x0s, horizon=2.0, ts=0.02)
    model, _ = fit_koopman_model(data, FitConfig(n_eig=[3, 3, 3], optimize=False))
    # lift matrix from basis points
    W = np.column_stack([model.lift(0.05 * e) / 0.05 for e in np.eye(3)])
    x = np.array([0.03, -0.02, 0.04])
    np.testing.assert_allclose(model.lift(x), W @ x, atol=1e-6)
    J = lift_jacobian(model, x, method="mls", k_neighbors=60)
    np.testing.assert_allclose(J, W, atol=1e-4)


def test_fd_jacobian_vector_step():
    """Per-dimension FD steps must equal the scalar-step result when all
    entries agree, and stay finite with anisotropic steps."""
    A = np.array([[0.0, 2.0], [-0.8, 2.0]])
    sys2 = linear_system(A)
    data = generate_data(
        sys2, circle_initial_conditions(0.2, 24), horizon=2.0, ts=0.01
    )
    model, _ = fit_koopman_model(data, FitConfig(n_eig=4, optimize=False))
    x = np.array([0.07, -0.03])
    J_scalar = lift_jacobian(model, x, method="fd", h=1e-3, order=2)
    J_vec = lift_jacobian(model, x, method="fd", h=np.array([1e-3, 1e-3]), order=2)
    np.testing.assert_allclose(J_vec, J_scalar)
    J_aniso = lift_jacobian(model, x, method="fd", h=np.array([1e-3, 5e-3]), order=2)
    assert np.all(np.isfinite(J_aniso))


# --------------------------------------------------------------------------
# Korda-Mezic constant-B regression
# --------------------------------------------------------------------------

def test_constant_b_regression_recovers_true_b():
    """On a linear plant xdot = A x + B u the lift is linear (Phi = W x),
    the true lifted input matrix is W B and is constant - the forced
    one-step regression must recover it from data."""
    A = np.array([[0.0, 2.0], [-0.8, 2.0]])
    B = np.array([[0.0], [1.0]])
    sys2 = linear_system(A)
    data = generate_data(
        sys2, circle_initial_conditions(0.2, 24), horizon=2.0, ts=0.01
    )
    model, _ = fit_koopman_model(data, FitConfig(n_eig=4, optimize=False))
    W = np.column_stack([model.lift(0.05 * e) / 0.05 for e in np.eye(2)])

    forced = ContinuousSystem(f=lambda x: A @ x, n=2)
    rng = np.random.default_rng(5)
    ts, K = 0.01, 40
    trajs = []
    for _ in range(6):
        x = 0.1 * rng.normal(size=2)
        u = rng.uniform(-1.0, 1.0, K)
        X = [x.copy()]
        for uk in u:
            step = ContinuousSystem(
                f=lambda xx, _u=uk: A @ xx + (B @ [_u]), n=2
            )
            x = step.simulate(x, np.array([0.0, ts]))[-1]
            X.append(x.copy())
        trajs.append((np.array(X), u))
    B_hat = fit_constant_b_forced(model, trajs, ts)
    np.testing.assert_allclose(B_hat, W @ B, atol=5e-3)

    # and forced prediction with the recovered B is accurate
    X, u = trajs[0]
    xp = predict_forced(model, B_hat, X[0], u, ts)
    np.testing.assert_allclose(xp, X, atol=1e-3)


def test_knn_b_field_recovers_constant_field():
    """On a linear plant the true lifted B is constant; the kNN smoother
    must reproduce it (weighted average of near-identical samples)."""
    from lpvkoopman import fit_b_knn

    A = np.array([[0.0, 2.0], [-0.8, 2.0]])
    B = np.array([0.0, 1.0])
    sys2 = linear_system(A)
    sys2.g = lambda x: B if np.ndim(x) == 1 else np.tile(B[:, None], (1, np.shape(x)[1]))
    data = generate_data(
        sys2, circle_initial_conditions(0.2, 24), horizon=2.0, ts=0.01
    )
    model, _ = fit_koopman_model(data, FitConfig(n_eig=4, optimize=False))
    W = np.column_stack([model.lift(0.05 * e) / 0.05 for e in np.eye(2)])
    rng = np.random.default_rng(7)
    pts = 0.15 * rng.normal(size=(80, 2))
    bknn = fit_b_knn(model, sys2.g, pts, k=15, f=sys2.f)
    for x in [np.array([0.02, 0.03]), np.array([-0.05, 0.01])]:
        np.testing.assert_allclose(bknn(x)[:, 0], W @ B, atol=1e-4)


def test_predict_forced_clips_bmap_state():
    """x_clip must saturate the state fed to a callable bmap."""
    A = np.array([[0.0, 2.0], [-0.8, 2.0]])
    sys2 = linear_system(A)
    data = generate_data(
        sys2, circle_initial_conditions(0.2, 24), horizon=2.0, ts=0.01
    )
    model, _ = fit_koopman_model(data, FitConfig(n_eig=4, optimize=False))
    seen = []

    def bmap(x):
        seen.append(x.copy())
        return np.zeros((model.N, 1))

    lo, hi = -0.01 * np.ones(2), 0.01 * np.ones(2)
    predict_forced(model, bmap, np.array([0.2, 0.0]), np.zeros(5), 0.01, x_clip=(lo, hi))
    assert all(np.all(s <= hi + 1e-12) and np.all(s >= lo - 1e-12) for s in seen)


# --------------------------------------------------------------------------
# end-to-end smoke on the extended system
# --------------------------------------------------------------------------

def test_extended_end_to_end_smoke():
    """Tiny extended fit: autonomous prediction at a held-out level within
    a loose tolerance, forced rollout finite and tracking x3 exactly."""
    sysna = vdp_nonaffine()
    levels = np.linspace(-0.8, 0.8, 5)
    data = make_extended_training_data(
        sysna, levels, n_traj_per_level=8, horizon=4.0, ts=0.02
    )
    model, _ = fit_koopman_model(
        data,
        FitConfig(n_eig=[10, 10, 1], constant_components=(2,), maxiter=200),
    )
    # autonomous check at a held-out level, near the training circles
    u = 0.2
    xeq = equilibrium(sysna, u)
    x0 = xeq + np.array([0.1, 0.0])
    t = np.arange(51) * 0.02
    xt = frozen(sysna, u).simulate(x0, t)
    xp = model.predict(np.array([*x0, u]), t)
    err = np.linalg.norm(xp[:, :2] - xt) / np.linalg.norm(xt)
    assert err < 0.2
    # forced rollout: x3 = u integrates v exactly through the lift
    v = 0.5 * np.ones(20)
    xf = predict_forced(model, np.zeros((model.N, 1)), np.array([*x0, u]), v, 0.02)
    # the x3 row of B is exact (dPhi3/dx3 = 1): with B = 0 the model must
    # NOT move x3, confirming x3 only responds through B
    np.testing.assert_allclose(xf[:, 2], u, atol=1e-8)
    A = model.A_continuous()
    Ad, Gam = discretize(model, 0.02)
    assert np.all(np.isfinite(Ad)) and np.all(np.isfinite(Gam))
