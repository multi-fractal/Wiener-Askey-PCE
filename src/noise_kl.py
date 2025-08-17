import numpy as np
from scipy.spatial.distance import cdist

# -------------------------
# KL on domain (any phys_dim)
# -------------------------
def compute_kl(domain_points, corr_length, variance, n_modes, jitter=1e-12):
    D = cdist(domain_points, domain_points, metric='euclidean')
    C = variance * np.exp(-D / corr_length)
    C += jitter * np.eye(C.shape[0])
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx][:n_modes]
    eigvecs = eigvecs[:, idx][:, :n_modes]
    eigvals = np.maximum(eigvals, 0.0)
    return eigvals, eigvecs

def field_from_kl(ksi_small, eigvals, eigvecs):
    sqrt_l = np.sqrt(eigvals)
    scaled = ksi_small * sqrt_l[None, :]      # (N_samples, n_modes)
    U = scaled @ eigvecs.T                   # (N_samples, n_domain)
    return U
