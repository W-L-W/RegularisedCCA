import pandas as pd
from itertools import product

covs = ['suo_sp_rand','powerlaw']
K_dict = {'suo_sp_rand':2, 'powerlaw':4}
algos = ['ridge','wit','suo','gglasso']
cols = ['cov_type','algo','ps','ns']#,'K','folds']

df_mini_test = pd.DataFrame([['suo_sp_rand','wit',[30],[10]]],columns=cols)

df_vary_n = pd.DataFrame(product(covs,algos,[[150]],[[20,40,60,100,150,200,300,500,1000]]),columns = cols)

df_comb = pd.concat([df_mini_test,df_vary_n]).reset_index().drop('index',axis=1)
df_comb['K'] = df_comb['cov_type'].apply(lambda cov: K_dict[cov])
df_comb['folds'] = 5


df_comb.to_csv('param_df.csv')
