import numpy as np
import pandas as pd
import os
from expmts.input_scripts.hpc.hpc_base import is_pboot

from src.scaffold.io_preferences import output_folder, save_mplib
from src.scaffold.synthetic import PenObjs, select_values, vary_n_best_rows
from hpc_base import get_row, create_mvn

import matplotlib.pyplot as plt

exp_idx = 12  #12 is pboot_nm_ggl_cvm3, ridge, many n
recompute = False


row = get_row(exp_idx)
print(row['cov_type'])

mvn = create_mvn(row)
folds = row['folds']
K = row['K']

# may want to use a subset of ns
ns = [20,40,60]
n_descr = 'few_n'
# make so can read from file system what has been computed already, or have an assertion that it has been computed
rss = range(0,32)

algo_color_dict = {
    'ridge':'grey',
    'wit':'red',
    'suo':'orange',
    'gglasso':'green'
}
algo_descr = 'all_algos'
# algos = ['ridge', 'wit']
# colours = ['grey', 'red']
#['ridge','wit','suo','gglasso'],['grey','red','orange','green']

fig_file_name = f"vary_n_{row['cov_type']}_{n_descr}_{algo_descr}"


pen_objs = PenObjs({'wt_u1':'min',
                    'vt_u1':'min',
                    'rho1':'max',
                    'rho1_cv':'max'}
)

def processed_df(algos):
    rows = dict()
    for algo in algos:
        rows[algo] = vary_n_best_rows(algo,mvn,folds,K,ns,rss,pen_objs)
    return rows

def load_best_rows(algos):
    import pickle

    path_stem = mvn.path_stem
    folder = output_folder(path_stem,mode='processed')
    file_name = folder + 'best_rows.pkl'

    if not os.path.exists(file_name) or recompute:
        rows = processed_df(algos) # the only heavyish computation step! # ,'suo','gglasso' add in if appropriate
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, 'wb') as f:
            pickle.dump(rows, f)
        #rows.to_csv(file_name)
    else:
        #rows = pd.read_csv(file_name,index_col=0)
        with open(file_name, 'rb') as f:
            rows = pickle.load(f)
    return rows


def make_plot(algo_color_dict):
    rows = load_best_rows(algo_color_dict.keys())
    print(rows.keys())
    
    fig,axs = plt.subplots(ncols=3,figsize=(12,6))

    for idx,met in enumerate(['wt_u1','vt_u1','rho1']):
        ax=axs[idx]
        ax.set_title(met)
        for algo,color in algo_color_dict.items():
            df_met = select_values(rows[algo],met)
            for objv,ls in zip(['rho1','rho1_cv'],[':','-']):
                label = algo + '+' + objv
                (df_met.xs(objv,axis=1,level=1)
                .plot(y='mean',label=label,ax=ax,ls=ls,color=color,
                    marker='.')
                )
    
    return fig, axs

if __name__ == "__main__":
    fig, axs = make_plot(algo_color_dict)
    save_mplib(fig,fig_file_name)