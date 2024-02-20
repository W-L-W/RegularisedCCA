from src.scaffold.incoming import get_pens
from src.scaffold.wrappers import get_dataset, get_cv_obj_from_data, compute_everything

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
for algo in algos:
    print(f'Fitting {algo}')
    cv_obj = get_cv_obj_from_data(data,algo)
    pens = get_pens(algo, data, mode='debug')
    print(pens)
    compute_everything(cv_obj,pens)


# then make all the plots
