# This is to be run via the command line with arguments
# rs: random seed, integer
# exp_idx: index of the experiment in the param_df.csv file

import numpy as np
import pandas as pd
import argparse

from src.algos import get_pens
from src.scaffold.synthetic import synth_setting, pboot_setting, one_seed_fit
from param_df_gen import is_pboot, param_df_file_name

exp_idx = 0
rs = 0

# # Argument Parsing
# parser = argparse.ArgumentParser(description='Get pareto plot data')
# parser.add_argument('random_seed', metavar='rs', type=int, nargs=1,
#                     help='random seed for data generation')
# parser.add_argument('exp_idx', metavar='ei', type=int, nargs=1,
#                                         help="experiment index see 'param_df_gen.py'")
# args = parser.parse_args()
# [rs] = args.random_seed
# [exp_idx] = args.exp_idx

# Extract parameters from corresponding row of param_df
param_df = pd.read_csv(param_df_file_name,index_col=0)
row = param_df.loc[exp_idx,:]
print(row)

cov_type = row['cov_type']
algo = row['algo']
ns = eval(row['ns'])
dims = eval(row['dims'])
K = row['K']
folds = row['folds']

if is_pboot(cov_type):
    create_setting = lambda n: pboot_setting(cov_type, algo, n, folds, K)
else:
    p, q = dims['p'], dims['q']
    create_setting = lambda n: synth_setting(cov_type, algo, p, q, n, folds, K)


for n in ns:
    n = int(n)
    pens = get_pens(algo,n)
    setting = create_setting(n)
    one_seed_fit(setting,rs,pens)
 