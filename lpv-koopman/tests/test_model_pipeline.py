import numpy as np
from scipy.linalg import expm

from lpvkoopman import (
    FitConfig,
    circle_initial_conditions,
    fit_koopman_model,
    generate_data,
    linear_system,
    vdp_scaled,
)
from lpvkoopman.metrics import relative_rmse_percent

A_LIN = np.array([[0.0, 2.0], [-0.8, 2.0]])  # VdP linearization (unstable focus)


def _linear_data():
    sys = linear_system(A_LIN)
    return sys, generate_data(
        sys, circle_initial_conditions(0.05, 16), horizon=2.0, ts=0.02
    )


def test_linear_system_recovered_to_machine_precision():
    """For xdot = A x the principal pair alone is an exact invariant
    subspace: lattice cost ~ 0 and prediction error ~ interpolation noise."""
    sys, data = _linear_data()
    model, rep = fit_koopman_model(data, FitConfig(n_eig=4, optimize=False))
    assert all(c["lattice_cost"] < 1e-16 for c in rep.components)

    lam = np.linalg.eigvals(A_LIN)
    np.testing.assert_allclose(
        sorted(rep.dmd_base.imag), sorted(lam.imag), atol=1e-8
    )

    t = np.arange(0.0, 1.0, 0.02)
    x0 = np.array([0.02, 0.01])
    x_true = np.array([expm(A_LIN * tk) @ x0 for tk in t])
    err = relative_rmse_percent(model.predict(x0, t), x_true)
    assert err < 1e-6


def test_propagate_matches_matrix_exponential():
    _, data = _linear_data()
    model, _ = fit_koopman_model(data, FitConfig(n_eig=8, optimize=False))
    A = model.A_continuous()
    z0 = np.linspace(-1.0, 1.0, model.N)
    for tk in [0.0, 0.13, 0.9]:
        np.testing.assert_allclose(
            model.propagate(z0, np.array([tk]))[0], expm(A * tk) @ z0, atol=1e-10
        )


def test_block_structure_and_real_output():
    _, data = _linear_data()
    model, _ = fit_koopman_model(data, FitConfig(n_eig=6, optimize=False))
    A = model.A_continuous()
    eigs_A = np.sort_complex(np.linalg.eigvals(A))
    eigs_spec = np.sort_complex(model.eigenvalues())
    np.testing.assert_allclose(eigs_A, eigs_spec, atol=1e-12)
    # C: one carrier per block
    assert model.C.shape == (2, model.N)
    assert np.all((model.C == 0) | (model.C == 1))
    x = model.predict(np.array([0.01, -0.02]), np.arange(0, 0.5, 0.1))
    assert x.dtype.kind == "f"  # real by construction


def test_lift_at_training_points_matches_table():
    """Linear interpolation evaluated at a triangulation vertex must return
    the tabulated value (consistency of the flat point ordering)."""
    _, data = _linear_data()
    model, _ = fit_koopman_model(data, FitConfig(n_eig=4, optimize=False))
    pts = data.flat_points()
    idx = [0, 57, 1234]
    Z = model.lift(pts[idx])
    np.testing.assert_allclose(Z, model.lifted_at_points[idx], atol=1e-9)


def test_fit_horizon_prefix():
    sys, data = _linear_data()
    model_full, _ = fit_koopman_model(data, FitConfig(n_eig=4, optimize=False))
    model_pre, _ = fit_koopman_model(
        data, FitConfig(n_eig=4, optimize=False, fit_horizon=1.0)
    )
    # same spectra (exact for linear data), tabulation along full horizon
    np.testing.assert_allclose(
        np.sort_complex(model_pre.eigenvalues()),
        np.sort_complex(model_full.eigenvalues()),
        atol=1e-6,
    )
    assert model_pre.points.shape == model_full.points.shape


def test_bidirectional_data_layout():
    sys = vdp_scaled()
    ics = circle_initial_conditions(0.3, 5)
    d = generate_data(sys, ics, horizon=0.5, ts=0.01, backward=0.4)
    assert d.t[0] == -0.4 and abs(d.t[-1] - 0.5) < 1e-12
    k0 = np.argmin(np.abs(d.t))
    np.testing.assert_allclose(d.X[:, k0, :], ics, atol=1e-12)
    np.testing.assert_allclose(np.diff(d.t), 0.01, atol=1e-12)


def test_vdp_end_to_end_smoke():
    """Small end-to-end VdP fit: interior prediction within a loose bound."""
    sys = vdp_scaled()
    data = generate_data(
        sys, circle_initial_conditions(0.05, 30), horizon=4.0, ts=0.02
    )
    model, rep = fit_koopman_model(data, FitConfig(n_eig=12, optimize=True))
    t = np.arange(0.0, 1.0, 0.02)
    x0 = np.array([-0.1382, 0.1728])  # the paper's showcase IC
    x_true = sys.simulate(x0, t)
    err = relative_rmse_percent(model.predict(x0, t), x_true)
    assert err < 15.0, f"interior VdP prediction error too large: {err:.2f}%"
