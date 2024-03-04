# note that code to run this file from command line is 
# /home/ww347/.conda/envs/myenv/bin/python3 /store/DPMMS/ww347/RegularisedCCA/expmts_in/pboot_single/nm/v0.py


from src.algos import get_pens
from src.scaffold.synthetic import load_pboot_mvn
from src.scaffold.wrappers import get_cv_obj_from_data, compute_everything
from src.plotting.oracle_panel import panel_plot
from src.plotting.comparison import row_plot, stab_row_plot

from src.scaffold.io_preferences import save_mplib

# the three commands to play with
dataset = 'microbiome'
regn = 'gglasso' # for parametric bootstrap; options 'gglasso', 'ridge', 'suo_ridge'
param_choice = 'cv' # how to choose the penalty
mode = 'run' # 'run' or 'debug'

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
mvn = load_pboot_mvn(dataset, regn, param_choice)
data = mvn.gen_data(rs,n)
print(data.folder_detail)

recompute = True
if recompute:
    for algo in algos:
        cv_obj = get_cv_obj_from_data(data,algo)
        compute_everything(cv_obj, get_pens(algo, data.n, mode))

    fig_panel = panel_plot(data)
    suffix = '_debug' if mode == 'debug' else ''
    save_mplib(fig_panel, f'pboot_panel_{abbrev[dataset]}_{abbrev[regn]}_{param_choice}'+ suffix)


fig, _, _ = row_plot(data, algos, lambda x: x**2, [0,2,4], y_label='r2sk(-cv)')
save_mplib(fig, f'pboot_traj_corr_{abbrev[dataset]}_{abbrev[regn]}_{param_choice}')

fig, _ = stab_row_plot(data, algos, ['vt_U', 'wt_U'], [1,3,5],)
save_mplib(fig, f'pboot_traj_stab_{abbrev[dataset]}_{abbrev[regn]}_{param_choice}')