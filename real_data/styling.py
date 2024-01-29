# Functions to determine biplots styling

# link to plotly doc page with different colorscales so can choose best one:
# plotly.com/python/builtin-colorscales/

import numpy as np
import pandas as pd

# BREASTDATA
############
def get_split_bd(labs):
    df = pd.Series(labs).str.split(pat=',',expand=True,n=1)
    if len(df.columns)==0:
        # warning became annoying, but might be nice to bring back later
        # print(f'Warning: empty dataframe, making template')
        return pd.DataFrame(columns=[0,1])
    else:
        return df
def chrom_2_num_bd(ser):
    return np.array([int(v) for v in ser])
def chrom_pos_2_num_bd(ser):
    return np.array([int(v.strip('k')) for v in ser])
 
max_chrom_pos = 243
bd_2d_style_fns = {'xmarkers': {'color': (lambda labs: chrom_2_num_bd(get_split_bd(labs)[0])),
                                'symbol': (lambda labs: chrom_pos_2_num_bd(get_split_bd(labs)[0])),
                                'cmin':1,'cmax':23},
                    'ymarkers': dict()}

bd_2d_style_fns2 = {'xmarkers': {'color': (lambda labs: chrom_pos_2_num_bd(get_split_bd(labs)[1])),
                                'symbol': (lambda labs: chrom_2_num_bd(get_split_bd(labs)[0])),
                                'cmin':0,'cmax':243,'size':7},
                    'ymarkers': {'color':'darkslategray','size':2}}

bd_3d_style_fns = {'xmarkers': {'color': (lambda labs: chrom_2_num_bd(get_split_bd(labs)[0])),
                                'cmin':1,'cmax':24, 'size':4},
                    'ymarkers': dict()}


# NUTRIMOUSE
############
def get_split_nm(labs):
    return pd.Series(labs).str.split(pat='.',expand=True)
def ser_2_colors_nm(ser):
    return np.array([int(v[1:]) for v in ser])
def ser_2_symbol_nm(ser):
    return np.array([int(v.strip('n')) for v in ser])

# to find good minimum and maximum values for the colorbar, previously ran
# labs = fullp_s.data.X_labs
# values = datasets.ser_2_colors_nm(datasets.get_split_nm(labs)[0])
# print(values.min(),values.max())

nm_2d_style_fns = {'xmarkers': {'color': (lambda labs: ser_2_colors_nm(get_split_nm(labs)[0])),
                                'symbol': (lambda labs: ser_2_symbol_nm(get_split_nm(labs)[1])),
                                'cmin':14,'cmax':22},
                    'ymarkers': dict()}
nm_3d_style_fns = {'xmarkers': {'color': (lambda labs: ser_2_colors_nm(get_split_nm(labs)[0])),
                                'cmin':14,'cmax':22},
                    'ymarkers': dict()}



# MICROBIOME
############

### numbers with which to divide the C0 number to get remainder for plot colour
n_div1 = 20
n_div2 = 19

def c0_2_int(c0):
    return int(c0[1:]) % n_div1
def c0_2_int2(c0):
    return int(c0[1:]) % n_div2
def labs_2_colors(c0s):
    return list(map(c0_2_int,c0s))
def labs_2_symbols(c0s):
    return list(map(c0_2_int2,c0s))

mb_2d_style_fns = {'ymarkers': {'color': (lambda labs: labs_2_colors(labs)),
                                'symbol': (lambda labs: labs_2_symbols(labs)),
                                'cmin':0,'cmax':n_div1,
                                'size':7},
                    'xmarkers': {'color':'LightSeaGreen','size':2}}
mb_3d_style_fns = {'ymarkers': {'color': (lambda labs: labs_2_colors(labs)),
                                'cmin':0,'cmax':n_div1, 'size':5},
                    'xmarkers': dict()}


# custom style functions for kumar's annotations
# amino acid metabolism
AAMet = ['K02804','K00975','K03431']
# ABC transporters
ABC = ['K10542','K15582','K02057','K10540','K01997','K15580','K02072','K06901','K01995','K02030','K10112']
# lipid metabolism
LMet = ['C00836','K00605','C08362','K02536']

# define function to get colors for each of these;
# amino acids: blue, ABC: red, lipids: green; grey otherwise
def get_color_kumar(lab):
    if lab in AAMet:
        return 'blue'
    elif lab in ABC:
        return 'red'
    elif lab in LMet:
        return 'green'
    else:
        if lab[0]=='K':
            return 'darkgrey'
        if lab[0]=='C':
            return 'black'
        else:
            print(f"warning - label didn't start with K or C: {lab}")
    
# define function to get sizes for each of these; size 6 if in AAMet, ABC or LMet; size 2 otherwise
def get_size_kumar(lab):
    # define union of the three lists
    union = AAMet + ABC + LMet
    if lab in union:
        return 12
    else:
        return 6 
    
# define custom style functions, colors only, using above functions 
# (same for 2d and 3d so only define once)
# also same function applied to both x and y variables
kumar_style_fns = {'xmarkers': {'color': (lambda labs: list(map(get_color_kumar,labs))),
                                'size': (lambda labs: list(map(get_size_kumar,labs)))},
                    'ymarkers': {'color': (lambda labs: list(map(get_color_kumar,labs))),
                                'size': (lambda labs: list(map(get_size_kumar,labs)))}}

