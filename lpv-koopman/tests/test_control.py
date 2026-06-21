import numpy as np
import pytest
from scipy.linalg import expm
from scipy.integrate import solve_ivp

from lpvkoopman import (
    FitConfig,
    circle_initial_conditions,
    fit_koopman_model,
    generate_data,
    linear_system,
)
from lpvkoopman.gradients import lift_jacobian, pde_residual
from lpvkoopman.lpv import BMap, discretize, predict_forced
from lpvkoopman.mpc import KoopmanMPC, MPCConfig, run_closed_loop
from lpvkoopman.systems import ContinuousSystem

A_LIN = np.array([[0.0, 2.0], [-0.8, 2.0]])  # unstable focus (VdP linearization)
B_LIN = np.array([0.0, 1.0])


@pytest.fixture(scope="module")
def linear_model():
    """Koopman model of the autonomous linear system; for xdot = A x the
    lift is exactly linear, so every control-side quantity has a known
    closed form to test against."""
    sys = linear_system(A_LIN)
    sys.g = lambda x: B_LIN if np.ndim(x) == 1 else np.tile(B_LIN[:, None], (1, np.shape(x)[1]))
    data = generate_data(sys, circle_initial_conditions(0.2, 24), horizon=2.0, ts=0.01)
    model, _ = fit_koopman_model(data, FitConfig(n_eig=4, optimize=False))
    return sys, model


def _exact_lift_matrix(model):
    """W with Phi(x) = W x, recovered from the model at two basis points.

    Valid because the linear system's eigenfunctions are linear; checked
    against a third point inside the test."""
    e1 = model.lift(np.array([0.05, 0.0])) / 0.05
    e2 = model.lift(np.array([0.0, 0.05])) / 0.05
    return np.column_stack([e1, e2])  # (N, 2)


def test_mls_jacobian_exact_for_linear_lift(linear_model):
    sys, model = linear_model
    W = _exact_lift_matrix(model)
    # linearity sanity: lift of a third point
    x3 = np.array([0.03, -0.04])
    np.testing.assert_allclose(model.lift(x3), W @ x3, atol=1e-8)
    for x in [np.array([0.05, 0.02]), np.array([-0.1, 0.08])]:
        J = lift_jacobian(model, x, method="mls")
        np.testing.assert_allclose(J, W, atol=1e-6)


def test_fd_jacobian_matches_mls(linear_model):
    sys, model = linear_model
    x = np.array([0.07, -0.03])
    J_mls = lift_jacobian(model, x, method="mls")
    J_fd = lift_jacobian(model, x, method="fd", h=1e-3, order=4)
    np.testing.assert_allclose(J_fd, J_mls, atol=1e-5)


def test_pde_residual_small_on_linear(linear_model):
    sys, model = linear_model
    rng = np.random.default_rng(0)
    X = 0.15 * rng.standard_normal((20, 2))
    r = pde_residual(model, sys.f, X)
    assert np.median(r) < 1e-5


def test_discretize_matches_expm(linear_model):
    sys, model = linear_model
    ts = 0.013
    Ad, Gam = discretize(model, ts)
    A = model.A_continuous()
    np.testing.assert_allclose(Ad, expm(A * ts), atol=1e-12)
    # exact identity for invertible A: Gamma = A^{-1} (expm(A ts) - I)
    exact = np.linalg.solve(A, expm(A * ts) - np.eye(model.N))
    np.testing.assert_allclose(Gam, exact, atol=1e-12)


def test_forced_prediction_exact_on_linear(linear_model):
    """LPV rollout (which is LTI here) must reproduce the true forced
    response of the linear plant to discretization accuracy."""
    sys, model = linear_model
    bmap = BMap(model, sys.g)
    B0 = bmap(np.array([0.05, 0.05]))
    # B(x) constant for a linear lift
    np.testing.assert_allclose(B0, bmap(np.array([-0.1, 0.02])), atol=1e-6)

    ts = 0.01
    K = 80
    rng = np.random.default_rng(1)
    u = rng.uniform(-1, 1, K)
    x0 = np.array([0.1, -0.05])
    Xp = predict_forced(model, bmap, x0, u, ts)

    # ground truth: exact ZOH discretization of (A_LIN, B_LIN)
    Adp = expm(A_LIN * ts)
    Gp = np.linalg.solve(A_LIN, (Adp - np.eye(2)) @ B_LIN.reshape(2, 1))
    x = x0.copy()
    Xt = [x.copy()]
    for uk in u:
        x = Adp @ x + (Gp[:, 0] * uk)
        Xt.append(x.copy())
    Xt = np.array(Xt)
    err = np.linalg.norm(Xp - Xt) / np.linalg.norm(Xt)
    assert err < 1e-4, f"forced prediction error {err:.2e}"


def test_mpc_stabilizes_unstable_linear_plant(linear_model):
    sys, model = linear_model
    bmap = BMap(model, sys.g)
    mpc = KoopmanMPC(model, bmap, MPCConfig(horizon=60, Q=(1.0, 1.0), R=1e-3, S=1e-3))
    res = run_closed_loop(mpc, sys, np.array([0.15, -0.1]),
                          reference=lambda t: np.zeros(2), n_steps=400)
    assert np.linalg.norm(res.X[-1]) < 1e-3, f"final |x| = {np.linalg.norm(res.X[-1]):.2e}"
    assert np.all(res.U >= -1 - 1e-9) and np.all(res.U <= 1 + 1e-9)
    # open loop is unstable: same horizon without control must grow
    free = sys.simulate(np.array([0.15, -0.1]), np.arange(0, 2.5, 0.01))
    assert np.linalg.norm(free[-1]) > 1.0


def test_mpc_tracks_reachable_reference(linear_model):
    sys, model = linear_model
    bmap = BMap(model, sys.g)
    # track x1 = 0.1: steady state x2 = 0, u* = 0.8*0.1 = 0.08 (within bounds)
    mpc = KoopmanMPC(model, bmap, MPCConfig(horizon=60, Q=(10.0, 0.0), R=1e-4, S=1e-3))
    res = run_closed_loop(mpc, sys, np.array([0.0, 0.0]),
                          reference=lambda t: np.array([0.1, 0.0]), n_steps=300)
    assert abs(res.X[-1, 0] - 0.1) < 5e-3, f"x1 settled at {res.X[-1, 0]:.4f}"


def test_polynomial_b_field_linear(linear_model):
    """For a linear lift B(x) is constant: any-degree fit must reproduce it
    and the rollout must match the exact discretization."""
    from lpvkoopman.lpv import fit_b_field

    sys, model = linear_model
    rng = np.random.default_rng(3)
    pts = 0.25 * rng.standard_normal((150, 2))
    bf = fit_b_field(model, sys.g, pts, degree=3, f=sys.f)
    bmap = BMap(model, sys.g)
    B0 = bmap(np.array([0.05, 0.05]))
    for xq in [np.zeros(2), np.array([0.1, -0.1])]:
        np.testing.assert_allclose(bf(xq), B0, atol=1e-5)

    ts, K = 0.01, 50
    u = np.random.default_rng(4).uniform(-1, 1, K)
    x0 = np.array([0.08, 0.02])
    Xp = predict_forced(model, bf, x0, u, ts)
    Adp = expm(A_LIN * ts)
    Gp = np.linalg.solve(A_LIN, (Adp - np.eye(2)) @ B_LIN.reshape(2, 1))
    x = x0.copy()
    Xt = [x.copy()]
    for uk in u:
        x = Adp @ x + Gp[:, 0] * uk
        Xt.append(x.copy())
    err = np.linalg.norm(Xp - np.array(Xt)) / np.linalg.norm(Xt)
    assert err < 1e-4
