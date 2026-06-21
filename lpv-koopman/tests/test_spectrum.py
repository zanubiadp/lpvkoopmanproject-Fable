import numpy as np
import pytest

from lpvkoopman.spectrum import (
    Spectrum,
    from_complex,
    lattice,
    lattice_spectrum,
    select_conjugate_closed,
)


VDP_BASE = np.array([1 + 0.7745966692j, 1 - 0.7745966692j])


def test_pack_unpack_roundtrip():
    spec = Spectrum(pairs=[[0.3, 1.2], [-0.5, 4.0]], reals=[2.0, 0.0])
    spec2 = Spectrum.unpack(spec.pack(), spec.n_pairs)
    np.testing.assert_allclose(spec2.pairs, spec.pairs)
    np.testing.assert_allclose(spec2.reals, spec.reals)
    assert spec.size == 6


def test_unpack_flips_negative_beta():
    spec = Spectrum.unpack(np.array([0.1, -2.0]), 1)
    assert spec.pairs[0, 1] == 2.0


def test_from_complex_groups_pairs():
    lams = [1 + 2j, 3.0, 1 - 2j]
    spec = from_complex(lams)
    assert spec.n_pairs == 1 and spec.n_reals == 1
    np.testing.assert_allclose(sorted(spec.to_complex(), key=lambda z: z.imag),
                               sorted(lams, key=lambda z: np.imag(z)))


def test_from_complex_rejects_unpaired():
    with pytest.raises(ValueError):
        from_complex([1 + 2j, 3.0])


def test_lattice_counts_and_order():
    lams = lattice(VDP_BASE, 3)
    # C(2+3, 3) = 10 distinct integer combinations with degree <= 3
    assert len(lams) == 10
    # zero is last
    assert lams[-1] == 0
    # conjugate-closed
    s = set(np.round(lams, 9))
    assert all(np.round(np.conj(l), 9) in s for l in lams)


@pytest.mark.parametrize("count", [2, 4, 6, 8, 10])
def test_lattice_spectrum_budget_and_closure(count):
    spec = lattice_spectrum(VDP_BASE, count)
    assert spec.size == count
    # principal pair always included first
    np.testing.assert_allclose(spec.pairs[0], [1.0, 0.7745966692], atol=1e-9)


def test_lattice_spectrum_n10_matches_paper_set():
    """N_i = 10 must reproduce the full degree-<=3 lattice used by
    Korda & Mezic for their N = 20 Van der Pol model."""
    spec = lattice_spectrum(VDP_BASE, 10)
    got = sorted(np.round(spec.to_complex(), 6).tolist(), key=lambda z: (z.real, z.imag))
    expected = sorted(
        np.round(lattice(VDP_BASE, 3), 6).tolist(), key=lambda z: (z.real, z.imag)
    )
    assert got == expected


def test_select_conjugate_closed_prefers_pair_over_zero():
    lams = lattice(VDP_BASE, 1)  # {lam, conj(lam), 0}
    spec = select_conjugate_closed(lams, 2)
    assert spec.n_pairs == 1 and spec.n_reals == 0
