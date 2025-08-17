import numpy as np
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

# -------------------------
# Covariance helpers
# -------------------------
def analytical_exponential_cov(r, sigma2, corr_length):
    return sigma2 * np.exp(-r / corr_length)

def pairwise_cov_matrix_from_ensemble(U):
    U = np.asarray(U)
    R, N = U.shape
    mean_U = U.mean(axis=0)
    mean_prod = (U.T @ U) / float(R)
    cov = mean_prod - np.outer(mean_U, mean_U)
    return cov

def covariance_vs_distance(domain, cov_mat, n_bins=40, max_dist=None):
    n_domain = domain.shape[0]
    dists = pdist(domain)
    iu = np.triu_indices(n_domain, k=1)
    cov_pairs = cov_mat[iu]
    if max_dist is None:
        max_dist = dists.max()
    bins = np.linspace(0.0, max_dist, n_bins+1)
    centers = 0.5*(bins[:-1]+bins[1:])
    cov_means = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    inds = np.digitize(dists, bins) - 1
    for k in range(n_bins):
        mask = inds == k
        counts[k] = mask.sum()
        if counts[k] > 0:
            cov_means[k] = cov_pairs[mask].mean()
    var_mean = np.mean(np.diag(cov_mat))
    return centers, cov_means, counts, var_mean

def compare_covariances(domain, U_true_ensemble, pce, ksi_test, sigma2, corr_length, n_bins=40, savepath=None):
    cov_true = pairwise_cov_matrix_from_ensemble(U_true_ensemble)
    r, cov_true_vals, _, var0_true = covariance_vs_distance(domain, cov_true, n_bins=n_bins)
    U_pce = pce.predict_at_ksi(ksi_test)
    cov_pce = pairwise_cov_matrix_from_ensemble(U_pce)
    _, cov_pce_vals, _, var0_pce = covariance_vs_distance(domain, cov_pce, n_bins=n_bins)
    cov_anal = analytical_exponential_cov(r, sigma2, corr_length)

    plt.figure(figsize=(8,5))
    plt.plot(r, cov_anal, label='Analytical exp', linewidth=2)
    plt.plot(r, cov_true_vals, 'o-', label=f'KL empirical (R={U_true_ensemble.shape[0]})')
    plt.plot(r, cov_pce_vals, 's--', label=f'PCE empirical (M={ksi_test.shape[0]})')
    plt.xlabel('Distance r'); plt.ylabel('Covariance C(r)')
    plt.legend(); plt.grid(True)
    if savepath: plt.savefig(savepath, dpi=150)
    plt.show()
    print("Var0 true:", var0_true, " Var0 pce:", var0_pce)
    return {'r': r, 'cov_anal': cov_anal, 'cov_true': cov_true_vals, 'cov_pce': cov_pce_vals}

# -------------------------
# Vizualization
# -------------------------
def visualize_field(domain, field_vals, phys_dim, title=None, savepath=None):
    if phys_dim == 1:
        plt.figure(figsize=(10,4))
        plt.plot(domain[:,0], field_vals, '-')
        if title: plt.title(title)
        if savepath: plt.savefig(savepath, dpi=150)
        plt.show()
    elif phys_dim == 2:
        x,y = domain[:,0], domain[:,1]
        plt.figure(figsize=(6,5))
        plt.tricontourf(x,y,field_vals, levels=30)
        plt.colorbar()
        if title: plt.title(title)
        if savepath: plt.savefig(savepath, dpi=150)
        plt.show()
    else:
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        x,y,z = domain[:,0], domain[:,1], domain[:,2]
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(x,y,z, c=field_vals, s=12)
        fig.colorbar(sc)
        if title: ax.set_title(title)
        if savepath: plt.savefig(savepath, dpi=150)
        plt.show()
