from src.scaffold.incoming import get_pens
from src.scaffold.synthetic import load_pboot_mvn
from src.scaffold.wrappers import get_cv_obj_from_data, compute_everything
from src.plotting.oracle_panel import panel_plot

from plots.saving import save_mplib

# the three commands to play with
dataset = 'microbiome'
regn = 'gglasso' # for parametric bootstrap; options 'gglasso', 'ridge', 'suo_ridge'
regn_mode = 'cvm3' # how to choose the penalty

abbrev = {'nutrimouse': 'nm',
          'breastdata': 'bd',
          'microbiome': 'mb',
          'gglasso': 'ggl',
          'ridge': 'rdg',
          'suo_ridge': 'suoR'
           }

nsamples = {'nutrimouse': 40,
            'microbiome': 458,
            }


# run once at start of session
algos = ['wit','suo','gglasso','ridge']
# parameters for the visualisations
rs = 0
n = nsamples[dataset] # number of samples in the true dataset
# create object for mvn and the data from the given random seed
mvn = load_pboot_mvn(dataset, regn, regn_mode)
data = mvn.gen_data(rs,n)
print(data.folder)

for algo in algos:
    cv_obj = get_cv_obj_from_data(data,algo)
    compute_everything(cv_obj, get_pens(algo, data, mode='run'))

fig_panel = panel_plot(data)
save_mplib(fig_panel, f'pboot_{abbrev[dataset]}_ggl_cvm3_panel')

