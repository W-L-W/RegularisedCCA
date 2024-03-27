from src.scaffold.synthetic import one_seed_fit
from expmts.input_scripts.hpc.hpc_base import get_row, setting_creator, pen_creator

# manually for debugging
rs = 10
exp_idx = 12

n=20

# Everything else
row = get_row(exp_idx)
create_setting = setting_creator(row)
create_pens = pen_creator(row, mode='debug') # 'debug' to fit smaller number of penalty parameters

pens = create_pens(n)
setting = create_setting(n)
one_seed_fit(setting,rs,pens)
