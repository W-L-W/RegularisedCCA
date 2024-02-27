# This is to be run via the command line with arguments
# rs: random seed, integer
# exp_idx: index of the experiment in the param_df.csv file

# Note I have the subm script on the hpc cluster only
# To run this script, navigate to it's directory and then run:
# `sbatch subm $exp_idx` 
# where $exp_idx is integer experiment index, as in this file

import argparse

from src.scaffold.synthetic import one_seed_fit
from expmts.input_scripts.hpc.hpc_base import get_row, setting_creator, pen_creator

# exp_idx = 0
# rs = 0

# Argument Parsing
parser = argparse.ArgumentParser(description='Get pareto plot data')
parser.add_argument('random_seed', metavar='rs', type=int, nargs=1,
                    help='random seed for data generation')
parser.add_argument('exp_idx', metavar='ei', type=int, nargs=1,
                                        help="experiment index see 'param_df_gen.py'")
args = parser.parse_args()
[rs] = args.random_seed
[exp_idx] = args.exp_idx


# Everything else
row = get_row(exp_idx)
create_setting = setting_creator(row)
create_pens = pen_creator(row)
ns = eval(row['ns'])

for n in ns:
    n = int(n)
    pens = create_pens(n)
    setting = create_setting(n)
    one_seed_fit(setting,rs,pens)
 
