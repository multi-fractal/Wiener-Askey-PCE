import os
import numpy as np
import scipy.special as sp
from itertools import product
from math import factorial

from utils import compare_covariances, visualize_field
from noise_kl import compute_kl, field_from_kl

# -------------------------
# Sampling ksi for different polynomial families
# -------------------------
def sample_ksi(N_samples, n_ksi, basis_type='hermite', kwargs=None):
    kwargs = kwargs or {}
    t = basis_type.lower()
    if t == 'hermite':
        return np.random.randn(N_samples, n_ksi)
    if t == 'legendre':
        return np.random.uniform(-1.0, 1.0, size=(N_samples, n_ksi))
    if t == 'laguerre':
        scale = kwargs.get('scale', 1.0)
        return np.random.exponential(scale=scale, size=(N_samples, n_ksi))
    if t == 'charlier':
        lam = kwargs.get('lam', 1.0)
        return np.random.poisson(lam=lam, size=(N_samples, n_ksi))
    raise ValueError("Unknown basis_type")

# -------------------------
# Univariate polynomial evaluators (probabilists' Hermite normalized)
# -------------------------
def eval_prob_hermite(n, x):
    return (2.0 ** (-n/2.0)) * sp.eval_hermite(n, x / np.sqrt(2.0))

def make_univariate_eval(poly_type='hermite', poly_kwargs=None):
    poly_kwargs = poly_kwargs or {}
    t = poly_type.lower()
    if t == 'hermite':
        def eval_uni(n, x):
            x = np.asarray(x)
            H = eval_prob_hermite(n, x)
            return H if n == 0 else H / np.sqrt(factorial(n))
        return eval_uni
    if t == 'legendre':
        return lambda n, x: sp.eval_legendre(n, x)
    if t == 'laguerre':
        return lambda n, x: sp.eval_laguerre(n, x)
    if t == 'jacobi':
        a = poly_kwargs.get('alpha', 0.0)
        b = poly_kwargs.get('beta', 0.0)
        return lambda n, x: sp.eval_jacobi(n, a, b, x)
    if t == 'charlier':
        a = poly_kwargs.get('a', 1.0)
        def eval_chr(n, x):
            xv = np.clip(np.round(x).astype(int), 0, None)
            if n == 0:
                return np.ones_like(xv, dtype=float)
            if n == 1:
                return xv.astype(float)/a - 1.0
            C0 = np.ones_like(xv, dtype=float)
            C1 = xv.astype(float)/a - 1.0
            for k in range(2, n+1):
                C2 = ((xv - (k-1) - a)/a) * C1 - ((k-1)/a) * C0
                C0, C1 = C1, C2
            return C1
        return eval_chr
    raise ValueError("Unknown poly_type")

# -------------------------
# Multi-index generator (total-degree)
# -------------------------
def total_degree_multiidxs(n_ksi, order):
    idxs = [idx for idx in product(range(order+1), repeat=n_ksi) if sum(idx) <= order]
    idxs.sort(key=lambda a: (sum(a), a))
    return idxs

# -------------------------
# Polynomial Chaos class
# -------------------------
class PolynomialChaos:
    def __init__(self, n_ksi, order, poly_type='hermite', poly_kwargs=None):
        self.n_ksi = n_ksi
        self.order = order
        self.poly_type = poly_type
        self.poly_kwargs = poly_kwargs or {}
        self.multi_idxs = total_degree_multiidxs(n_ksi, order)
        self.n_basis = len(self.multi_idxs)
        self.eval_uni = make_univariate_eval(poly_type, self.poly_kwargs)
        self.sample_ksi = None    # (N_train, n_ksi)
        self.coeffs = None       # (n_basis, n_domain)

    def set_sample_ksi(self, ksi):
        ksi = np.asarray(ksi)
        assert ksi.ndim == 2 and ksi.shape[1] == self.n_ksi
        self.sample_ksi = ksi

    def build_Psi(self, ksi=None):
        if ksi is None:
            ksi = self.sample_ksi
        assert ksi is not None
        N = ksi.shape[0]
        Psi = np.ones((self.n_basis, N))
        for j, alpha in enumerate(self.multi_idxs):
            val = np.ones(N, dtype=float)
            for d, deg in enumerate(alpha):
                if deg > 0:
                    val *= self.eval_uni(deg, ksi[:, d])
            Psi[j, :] = val
        return Psi

    def compute_coefficients(self, U_train):
        """
        U_train: (N_train, n_domain)
        """
        if self.sample_ksi is None:
            raise RuntimeError("set_sample_ksi before compute_coefficients")
        U_train = np.asarray(U_train)
        assert U_train.ndim == 2
        N = self.sample_ksi.shape[0]
        assert U_train.shape[0] == N, "Mismatch train sample size"
        Psi = self.build_Psi()                        # (n_basis, N)
        numerator = (Psi @ U_train) / float(N)        # (n_basis, n_domain)
        denom = (Psi**2).mean(axis=1)                 # (n_basis,)
        print("Diagnostics denom (first 12):", np.round(denom[:12],6))
        print("denom min/median/max:", np.round(denom.min(),6), np.round(np.median(denom),6), np.round(denom.max(),6))
        denom_safe = np.where(denom <= 0, 1e-16, denom)
        self.coeffs = numerator / denom_safe[:, None]
        return self.coeffs

    def predict_at_ksi(self, ksi_test):
        ksi_test = np.asarray(ksi_test)
        Psi_test = self.build_Psi(ksi_test)            # (n_basis, M)
        U_pred = Psi_test.T @ self.coeffs             # (M, n_domain)
        return U_pred

    def mean_field(self):
        if self.coeffs is None:
            raise RuntimeError("coeffs not computed")
        return self.coeffs[0, :]

# -------------------------
# Domain generation and visualization
# -------------------------
def generate_domain(phys_dim, total_points_number):
    if phys_dim == 1:
        x = np.linspace(-1.0, 1.0, total_points_number)
        return x.reshape(-1,1)
    if phys_dim == 2:
        side = int(np.sqrt(total_points_number))
        ax = np.linspace(-1.0, 1.0, side)
        X,Y = np.meshgrid(ax, ax)
        return np.column_stack([X.ravel(), Y.ravel()])
    if phys_dim == 3:
        side = int(round(total_points_number ** (1/3)))
        ax = np.linspace(-1.0, 1.0, side)
        X,Y,Z = np.meshgrid(ax, ax, ax)
        return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    raise ValueError("phys_dim must be 1,2,3")

# -------------------------
# Pipeline with train/test split
# -------------------------
def run_pipeline(
    phys_dim=1,
    total_points_number=300,
    n_ksi=6,
    pce_order=3,
    n_train_samples=10000,
    n_test_samples=10000,
    corr_length=1.0,
    variance=1.0,
    basis_type='hermite',
    basis_kwargs=None,
    plot_dir='figures_tt'
):
    os.makedirs(plot_dir, exist_ok=True)
    domain = generate_domain(phys_dim, total_points_number)
    print("Domain:", domain.shape)

    # KL with n_ksi modes
    eigvals, eigvecs = compute_kl(domain, corr_length=corr_length, variance=variance, n_modes=n_ksi)
    print("KL modes:", eigvals.shape, eigvecs.shape)

    # Generate training ksi and corresponding U (via KL)
    ksi_train = sample_ksi(n_train_samples, n_ksi, basis_type=basis_type, kwargs=basis_kwargs)
    U_train = field_from_kl(ksi_train, eigvals, eigvecs)    # (n_train_samples, n_domain)
    print("ksi_train:", ksi_train.shape, "U_train:", U_train.shape)

    # Build and train PCE
    pce = PolynomialChaos(n_ksi=n_ksi, order=pce_order, poly_type=basis_type, poly_kwargs=basis_kwargs)
    pce.set_sample_ksi(ksi_train)
    coeffs = pce.compute_coefficients(U_train)
    print("Coeffs shape:", coeffs.shape, "n_basis=", pce.n_basis)

    # Generate test ksi and evaluate ground truth + PCE
    ksi_test = sample_ksi(n_test_samples, n_ksi, basis_type=basis_type, kwargs=basis_kwargs)
    U_true_test = field_from_kl(ksi_test, eigvals, eigvecs)   # (n_test_samples, n_domain)
    U_pce_test = pce.predict_at_ksi(ksi_test)                  # (n_test_samples, n_domain)

    # Diagnostics: variance reproduction (avg over domain)
    var_true = np.mean(np.var(U_true_test, axis=0))
    var_pce = np.mean(np.var(U_pce_test, axis=0))
    print("Var true (avg over domain):", var_true)
    print("Var pce  (avg over domain):", var_pce)
    print("Ratio pce/true:", var_pce / (var_true + 1e-16))

    # Compare covariances
    comp = compare_covariances(domain, U_true_test, pce, ksi_test, sigma2=variance, corr_length=corr_length,
                               n_bins=60, savepath=os.path.join(plot_dir, "cov_comp.png"))

    # Visualize sample and mean
    pce_mean = pce.mean_field()
    idx0 = 0
    visualize_field(domain, U_true_test[idx0,:], phys_dim, title="True sample (KL)", savepath=os.path.join(plot_dir,"true_sample.png"))
    visualize_field(domain, pce_mean, phys_dim, title="PCE mean", savepath=os.path.join(plot_dir,"pce_mean.png"))
    visualize_field(domain, U_pce_test[idx0,:], phys_dim, title="PCE sample", savepath=os.path.join(plot_dir,"pce_sample.png"))

    # Save diagnostics
    np.savez(os.path.join(plot_dir, "diagnostics.npz"),
             eigvals=eigvals, coeffs=coeffs, var_true=var_true, var_pce=var_pce)
    print("Output saved to", plot_dir)
    return {
        'domain': domain, 'eigvals': eigvals, 'eigvecs': eigvecs,
        'ksi_train': ksi_train, 'U_train': U_train,
        'ksi_test': ksi_test, 'U_true_test': U_true_test, 'U_pce_test': U_pce_test,
        'pce': pce, 'comp': comp
    }

# -------------------------
# main
# -------------------------
def main():
    res = run_pipeline(
        phys_dim=1,
        total_points_number=300,
        n_ksi=10,
        pce_order=3,
        n_train_samples=15000,  # coefficients estimation
        n_test_samples=15000,   # covariance estimation
        corr_length=0.5,
        variance=1.0,
        basis_type='hermite',   # hermite | legendre | laguerre | charlier
        basis_kwargs=None,
        plot_dir='figures_train_test'
    )


if __name__ == "__main__":
    main()
