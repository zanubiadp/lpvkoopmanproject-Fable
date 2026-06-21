"""Eigenvalue handling: DMD estimates, lattice generation, and the canonical
real parametrization of a conjugate-closed spectrum.

A conjugate-closed set of continuous-time eigenvalues is stored as

    pairs : (P, 2) array of (alpha_p, beta_p), beta_p > 0, representing
            alpha_p +- i beta_p  (two eigenvalues each)
    reals : (R,)  array of real eigenvalues

so the total count is N = 2 P + R. This is the thesis Appendix-A
parametrization; it guarantees by construction that lifted predictions of
real states are real, and it halves the number of optimization variables
relative to optimizing unconstrained complex eigenvalues.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class Spectrum:
    """Conjugate-closed set of continuous-time eigenvalues."""

    pairs: Array  # (P, 2) rows (alpha, beta), beta > 0
    reals: Array  # (R,)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "pairs", np.asarray(self.pairs, dtype=float).reshape(-1, 2)
        )
        object.__setattr__(
            self, "reals", np.asarray(self.reals, dtype=float).reshape(-1)
        )
        if self.pairs.size and np.any(self.pairs[:, 1] <= 0):
            raise ValueError("pair imaginary parts must be strictly positive")

    @property
    def n_pairs(self) -> int:
        return self.pairs.shape[0]

    @property
    def n_reals(self) -> int:
        return self.reals.shape[0]

    @property
    def size(self) -> int:
        """Total number of (complex) eigenvalues represented."""
        return 2 * self.n_pairs + self.n_reals

    def to_complex(self) -> Array:
        """All eigenvalues as complex numbers (pairs first, + before -)."""
        lams = []
        for a, b in self.pairs:
            lams += [a + 1j * b, a - 1j * b]
        lams += list(self.reals.astype(complex))
        return np.array(lams, dtype=complex)

    # --- flat parameter vector for optimizers -------------------------------
    def pack(self) -> Array:
        """Flatten to [a_1, b_1, ..., a_P, b_P, r_1, ..., r_R]."""
        return np.concatenate([self.pairs.reshape(-1), self.reals])

    @staticmethod
    def unpack(theta: Array, n_pairs: int) -> "Spectrum":
        theta = np.asarray(theta, dtype=float).reshape(-1)
        pairs = theta[: 2 * n_pairs].reshape(n_pairs, 2).copy()
        # Identifiability: cos/sin columns are even/odd in beta, so flip
        # negative betas back to the canonical beta > 0 half-plane.
        pairs[:, 1] = np.abs(pairs[:, 1])
        pairs[:, 1] = np.maximum(pairs[:, 1], 1e-12)
        return Spectrum(pairs=pairs, reals=theta[2 * n_pairs :].copy())

    def __str__(self) -> str:
        terms = [f"{a:.4f}±{b:.4f}i" for a, b in self.pairs]
        terms += [f"{r:.4f}" for r in self.reals]
        return "{" + ", ".join(terms) + "}"


def from_complex(lams: Array, tol: float = 1e-9) -> Spectrum:
    """Group a conjugate-closed complex eigenvalue list into a Spectrum.

    Raises if the set is not conjugate-closed within ``tol``.
    """
    lams = list(np.asarray(lams, dtype=complex))
    pairs, reals = [], []
    while lams:
        lam = lams.pop(0)
        if abs(lam.imag) <= tol:
            reals.append(lam.real)
            continue
        # find conjugate partner
        match = None
        for k, other in enumerate(lams):
            if abs(other - np.conj(lam)) <= tol * max(1.0, abs(lam)):
                match = k
                break
        if match is None:
            raise ValueError(f"eigenvalue {lam} has no conjugate partner")
        lams.pop(match)
        pairs.append((lam.real, abs(lam.imag)))
    return Spectrum(
        pairs=np.array(pairs, dtype=float).reshape(-1, 2),
        reals=np.array(sorted(reals, reverse=True), dtype=float),
    )


def dmd_eigenvalues(X: Array, ts: float) -> Array:
    """Continuous-time eigenvalues of the one-step DMD operator.

    X : (M_t, M_s + 1, n) trajectory tensor.

    Builds the snapshot pairs (x_k, x_{k+1}) within each trajectory, fits the
    best linear map X2 = A X1 in the least-squares sense and returns
    log(eig(A)) / ts. For data concentrated near a hyperbolic fixed point
    this approximates the linearization spectrum.
    """
    X1 = X[:, :-1, :].reshape(-1, X.shape[2]).T  # (n, K)
    X2 = X[:, 1:, :].reshape(-1, X.shape[2]).T
    A_dmd, *_ = np.linalg.lstsq(X1.T, X2.T, rcond=None)
    mu = np.linalg.eigvals(A_dmd.T)
    return np.log(mu.astype(complex)) / ts


def lattice(base: Array, max_degree: int) -> Array:
    """Nonnegative-integer-combination lattice of base eigenvalues.

    lattice_d(L) = { sum_k a_k L_k : a_k >= 0 integers, sum_k a_k <= d }

    (Korda & Mezic 2020, eq. (54); includes 0 from the all-zero combination.)
    For a conjugate pair base the result is conjugate-closed. Returned in a
    deterministic priority order used for budget truncation: ascending total
    degree with complex pairs before real combinations of the same degree
    (conjugate partners adjacent, + first), and the zero eigenvalue (constant
    eigenfunction) last - it only ever models a DC offset, so it should be
    the first to go when the budget is tight.
    """
    base = np.asarray(base, dtype=complex).reshape(-1)
    p = len(base)

    combos: list[tuple[int, complex]] = []

    def rec(idx: int, left: int, acc: complex, deg: int) -> None:
        if idx == p:
            combos.append((deg, acc))
            return
        for a in range(left + 1):
            rec(idx + 1, left - a, acc + a * base[idx], deg + a)

    rec(0, max_degree, 0.0 + 0.0j, 0)

    # Deduplicate (e.g. repeated sums) with rounding-based keys.
    seen: dict[tuple[int, float, float], tuple[int, complex]] = {}
    for deg, lam in combos:
        key = (deg, round(lam.real, 12), round(lam.imag, 12))
        if key not in seen:
            seen[key] = (deg, lam)
    items = list(seen.values())
    items.sort(
        key=lambda dl: (
            abs(dl[1]) < 1e-14,  # lambda = 0 last
            dl[0],  # total degree
            abs(dl[1].imag) < 1e-12,  # pairs before reals within a degree
            -abs(dl[1].imag),
            -dl[1].imag,
        )
    )
    return np.array([lam for _, lam in items], dtype=complex)


def select_conjugate_closed(lams: Array, count: int, tol: float = 1e-9) -> Spectrum:
    """Pick the first ``count`` eigenvalues from an ordered lattice while
    keeping the selection conjugate-closed.

    Walks the (degree-ordered) lattice; complex eigenvalues are taken together
    with their conjugate partner, reals one at a time, until ``count`` is
    reached. If a final complex pair does not fit (one slot left), the next
    real eigenvalue is used instead.
    """
    lams = list(np.asarray(lams, dtype=complex))
    pairs: list[tuple[float, float]] = []
    reals: list[float] = []
    used = [False] * len(lams)

    def room() -> int:
        return count - (2 * len(pairs) + len(reals))

    for i, lam in enumerate(lams):
        if used[i] or room() <= 0:
            continue
        if abs(lam.imag) <= tol:
            reals.append(lam.real)
            used[i] = True
            continue
        if room() < 2:
            continue  # cannot fit a pair; keep scanning for a real
        # locate conjugate partner
        for k in range(i + 1, len(lams)):
            if not used[k] and abs(lams[k] - np.conj(lam)) <= tol * max(1.0, abs(lam)):
                pairs.append((lam.real, abs(lam.imag)))
                used[i] = used[k] = True
                break
    if room() != 0:
        raise ValueError(
            f"could not assemble a conjugate-closed set of {count} eigenvalues "
            f"from {len(lams)} candidates"
        )
    return Spectrum(
        pairs=np.array(pairs, dtype=float).reshape(-1, 2),
        reals=np.array(reals, dtype=float),
    )


def lattice_spectrum(base: Array, count: int) -> Spectrum:
    """Smallest-degree lattice spectrum with exactly ``count`` eigenvalues.

    Increases the lattice degree until at least ``count`` eigenvalues are
    available (Korda & Mezic choose d_lat with N_lat >= count), then selects a
    conjugate-closed subset in degree order.
    """
    for d in range(1, 40):
        lams = lattice(base, d)
        if len(lams) >= count:
            try:
                return select_conjugate_closed(lams, count)
            except ValueError:
                continue  # need one more degree to close conjugacy
    raise ValueError("lattice degree search exhausted")
