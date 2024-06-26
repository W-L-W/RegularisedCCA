from .cca import (
    cca_from_cov_mat,
    can_corr_subs,
    emp_cor_mat,
    true_cor_mat,
    emp_cov_mat,
    true_cov_mat,
    soft_threshold,
    suo_init_K,
    suo_init,
    data_from_covariance,
    oracle_cos2thetas,
)

from .linalg import (
    isPSD,
    get_near_psd,
    sin2theta,
    sin2theta_mult,
    sq_trigs,
    register_general,
    gram_schmidt,
    col_corr,
    threshold_corrs_mat_only,
    get_l1_mask,
)
