import numpy as np
import os

from src.scaffold.incoming import get_pens
from src.scaffold.wrappers import get_cv_obj_from_data, compute_everything, get_dataset

from real_data.loading import output_folder_real_data
# parameters for this script
# to move to kwargs later so that can be run from command line

dataset = 'nutrimouse'


# wit, i.e. sPLS, is not a genuine CCA algorithm, and gglasso scales too poorly to fit on the full breastdata set
algos_all = ['wit','suo','gglasso','ridge']
algos_CCA = ['suo','gglasso','ridge']
algos_bd = ['wit','suo','ridge']
algos_CCA_bd = ['suo','ridge']

algos_to_fit = {'nutrimouse': algos_all, 'microbiome': algos_all, 'breastdata': algos_bd}
algos = algos_to_fit[dataset]
data = get_dataset(dataset)

# first fit all the algorithms
# for algo in algos:
#     print(f'Fitting {algo}')
#     cv_obj = get_cv_obj_from_data(data,algo)
#     pens = get_pens(algo, data, mode='run')
#     print(pens)
#     compute_everything(cv_obj,pens)


# then process the output and make the plots
def get_pen_trios(algos, data):
    file_name = output_folder_real_data(dataset, mode='processed') + 'pen_trios.npz'
    print(file_name)
    # if file doesn't exist compute pen trio
    if not os.path.exists(file_name):
        def get_trio(scale): return float(scale)**np.array([-1,0,1])
        def get_best_pen(algo): return get_cv_obj_from_data(data,algo).get_best_pen('r2s5')
        pen_trios = {}
        for algo in algos:
            pen_trios[algo] = get_best_pen(algo) * get_trio(3)
        # save this dictionary to file
        # create the parent directory if required
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        np.savez(file_name, **pen_trios)
    else:
        pen_trios = np.load(file_name)

if __name__ == '__main__':
    get_pen_trios(algos, data)
    print('done')

# progress at lunch break: need to fix error on console about get_best_pen
        
        
