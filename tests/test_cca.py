import numpy as np
from src.utils.linalg import nsqrtm, sq_trigs
from src.utils.cca import cca_from_cov_mat
import src.utils.covs as covs

def test_spiked_covariance_model_population_recovery():
    p, q = 20, 40
    K = 8
    decay_ratio = 0.9; spike_size = 3
    Sig = covs.geom_corr_decay_sparse_weight_multi_spike(p, q, K, decay_ratio, spike_size, method='sparse', geom_param=0.9)

    U, V, R = cca_from_cov_mat(Sig, p, zero_cut_off=None)
    assert np.isclose(R[:K], [decay_ratio**k for k in range(1,K+1)]).all()

