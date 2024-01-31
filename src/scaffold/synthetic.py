# Think intention here was to copy over the block of code from oop_utils starting with 'Setting' class
# TODO work through that!
from src.scaffold.interface import get_dataset, bootstrap_params
from src.algos import first_K_ccs_lazy
from src.utils import gram_schmidt
import numpy as np


def gen_parametric_bootstrap_cov(dataset: str, regn='ridge', regn_mode = 'cv'):
    data = get_dataset(dataset)
    def demean_cols(M): return M - M.mean(axis=0)
    X,Y = list(map(demean_cols,[data.X,data.Y]))
    n,p = X.shape[:2]; q = Y.shape[1]
    #convert to joint data
    Z = np.block([X,Y])
    emp_cov = Z.T @ Z/n

    # get penalty
    params = bootstrap_params[(dataset,regn,regn_mode)]

    # estimate regularised covariance
    if regn == 'ridge':
        ridge_param = params['ridge']
        Sig = emp_cov +  ridge_param * np.identity(p+q)
    elif regn == 'gglasso':
        from gglasso.problem import glasso_problem
        pen_param = params['pen']
        reg_params = {'lambda1': pen_param}
        P = glasso_problem(emp_cov, n, reg_params = reg_params, latent = False, do_scaling = False)
        P.solve(solver_params={'max_iter':100})
        print('solved glasso problem')
        Sig = np.linalg.inv(P.solution.precision_)
    elif regn == 'suo_ridge':
        Xtil,Ytil = X/np.sqrt(n),Y/np.sqrt(n)
        pen_param, K, ridge_param = params['pen'], params['K'], params['ridge']
        taus = [pen_param]*K
        Ue,Ve,te0 = first_K_ccs_lazy(Xtil,Ytil,taus)
        De = np.diag(np.diag(Ue.T @ Xtil.T @ Ytil @ Ve)) # extract diagonal elements
        print(De)
        # np.sqrt(np.log(p) / n) * np.identity(p)
        Sigxx = Xtil.T @ Xtil +  ridge_param
        Sigyy = Ytil.T @ Ytil +  ridge_param

        #(Ue.T @ Xtil.T @ Ytil @ Ve)[:8,:8].round(2) # to check looks reasonable
        Ue = gram_schmidt(Ue,Sigxx)
        Ve = gram_schmidt(Ve,Sigyy)

        Sigxy = Sigxx @ Ue @ De @ Ve.T @ Sigyy
        Sigyx = Sigxy.T
        Sig = np.block([[Sigxx,Sigxy],[Sigyx,Sigyy]])
    else:
        raise NotImplementedError()
    
    return Sig

