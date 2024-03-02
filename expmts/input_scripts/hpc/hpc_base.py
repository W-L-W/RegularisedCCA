import pandas as pd


from src.algos import get_pens
from src.scaffold.synthetic import pboot_mvn, synth_mvn, Setting

# get parent directory of the current file
import os
dir_path_this_script = os.path.dirname(os.path.realpath(__file__)) + '/'
param_df_file_name = dir_path_this_script + 'param_df.csv'

# Extract parameters from corresponding row of param_df
def get_row(exp_idx: int):
    param_df = pd.read_csv(param_df_file_name,index_col=0)
    row = param_df.loc[exp_idx,:]
    return row


def is_pboot(cov_type: str):
    return 'pboot' in cov_type

def create_mvn(row):
    cov_type = row['cov_type']
    dims = row['dims']
    if is_pboot(cov_type):
        assert dims == 'pboot', 'pboot should have dims = "pboot"'
        mvn = pboot_mvn(cov_type)
    else:
        dims = eval(dims) # dims is a string representation of a dictionary
        p, q = dims['p'], dims['q']
        mvn = synth_mvn(cov_type, p, q)
    return mvn

def setting_creator(row):
    mvn = create_mvn(row)
    algo = row['algo']
    K = row['K']
    folds = row['folds']
    return lambda n: Setting(mvn, algo, n, folds, K)

def pen_creator(row, mode='run'):
    """mode: 'run' or 'debug' as per get_pens() in src/algos.py"""
    algo = row['algo']
    return lambda n: get_pens(algo,n, mode=mode)