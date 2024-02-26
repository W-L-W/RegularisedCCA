import pandas as pd
from itertools import product
from typing import Iterable

# get parent directory of the current file
import os
dir_path_this_script = os.path.dirname(os.path.realpath(__file__)) + '/'
param_df_file_name = dir_path_this_script + 'param_df.csv'

K_dict = {'suo_sp_rand':2, 
          'powerlaw':4, 
          'pboot_nm_ggl_cvm3': 6,
          'pboot_mb_ggl_cv': 6,
}

def is_pboot(cov_type: str):
    return 'pboot' in cov_type

def assign_csts(df: pd.DataFrame, cols: Iterable[str], vals: Iterable):
    for col, val in zip(cols, vals):
        df[col] = [val] * len(df)
    return df

covs = K_dict.keys()
covs_pboot = [k for k in K_dict.keys() if 'pboot' in k]
covs_synth = [k for k in K_dict.keys() if k not in covs_pboot]

algos = ['ridge','wit','suo','gglasso']
cols = ['cov_type','algo','dims','ns']#,'K','folds']

df_mini_test = pd.DataFrame([['suo_sp_rand','wit',{'p':10, 'q':10},[20]]],columns=cols)

df_synth = (pd.DataFrame(product(covs_synth,algos), columns = ['cov_type', 'algo'])
            .pipe(assign_csts, ['dims', 'ns'], [{'p':100, 'q':100}, [20,40,60,100,150,200,300,500,1000]])
)
df_pboot = (pd.DataFrame(product(covs_pboot,algos), columns = ['cov_type', 'algo'])
            .pipe(assign_csts, ['dims', 'ns'], ['pboot', [20,40,60,100,150,200,300,500,1000]])
)
df_comb = pd.concat([df_mini_test, df_synth, df_pboot]).reset_index().drop('index',axis=1)
df_comb['K'] = df_comb['cov_type'].apply(lambda cov: K_dict[cov])
df_comb['folds'] = 5


df_comb.to_csv(param_df_file_name)
