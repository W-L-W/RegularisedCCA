# Main implementational code to load datasets from csv files

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.scaffold.core import Data
from src.scaffold.incoming import base_ds, real_data_base_ds


# Shared helper functions
def demean_cols(M):
    return M - M.mean(axis=0)

def raw_data_filename(dataset: str, filename: str) -> str:
    raw_directory = real_data_base_ds + '/' + dataset + '/raw/'
    return raw_directory + filename

def pboot_filename(dataset: str, filename: str) -> str:
    pboot_directory = real_data_base_ds + '/' + dataset + '/pboot/'
    return pboot_directory + filename

# BREASTDATA
############

def get_breastdata():
    X0 = np.array(pd.read_csv(raw_data_filename('breastdata', 'dna_matrix.csv'), index_col=0).T)
    Y0 = np.array(pd.read_csv(raw_data_filename('breastdata', 'rna_matrix.csv'), index_col=0).T)

    X, Y = list(map(demean_cols, [X0, Y0]))

    dna_labels_full = pd.read_csv(raw_data_filename('breastdata', 'dna_labels.csv'), index_col=0)
    dna_labels = dna_labels_full.apply(lambda row: str(int(row['chrom'])) + ',' + str(int(row['nuc'] // 1000)) + 'k', axis=1)
    dna_labels.name = 'dna labels'

    rna_labels_full = pd.read_csv(raw_data_filename('breastdata', 'rna_labels.csv'), index_col=0)
    rna_labels = rna_labels_full['genename']
    rna_labels.name = 'rna_labels'

    data = Data(X, Y, folder_name=base_ds + 'breastdata', X_labs=dna_labels, Y_labs=rna_labels)
    return data


# NUTRIMOUSE
############
def get_nutrimouse():
    lipids = pd.read_csv(raw_data_filename('nutrimouse', 'lipid.csv'))
    genes = pd.read_csv(raw_data_filename('nutrimouse', 'gene.csv'))

    X0 = np.array(lipids)
    Y0 = np.array(genes)

    X,Y = list(map(demean_cols,[X0,Y0]))

    sc = StandardScaler()
    sc.fit(lipids)
    X = np.array(sc.transform(lipids))
    sc.fit(genes)
    Y = np.array(sc.transform(genes))

    X_labs = lipids.columns.rename('lipids')
    Y_labs = genes.columns.rename('genes')

    data = Data(X,Y,folder_name=base_ds+'nutrimouse',
               X_labs=X_labs,Y_labs=Y_labs)
    return data


# MICROBIOME
############
def get_microbiome():
    # note only considering prep_type `v1` for now; potentially to review
    prep_type='v1'
    X,Y,X_cols,Y_cols,index = prep_type_to_data(prep_type)
    X_cols.name = 'k0_KEGG'; Y_cols.name = 'met_KEGG'
    data = Data(X,Y,folder_name=base_ds+'kumar_oop/'+prep_type,
                   X_labs=X_cols,Y_labs=Y_cols)
    return data

## Data preparation and computation of estimates
def prep_type_to_data(prep_type):
    # transpose so that rows are shared as in data-matrix setup
    k0df = pd.read_csv(raw_data_filename('microbiome','ko_hmp2.csv'),index_col='KEGG').T
    metdf = pd.read_csv(raw_data_filename('microbiome','metabolites_hmp2.csv'),index_col='KEGG').T

    patients_in_both = k0df.index.intersection(metdf.index)

    k0df_shared = k0df.loc[patients_in_both]
    metdf_shared = metdf.loc[patients_in_both]

    def remove_cols_with_zeros(df): return df.loc[:,(df != 0).all(axis=0)]
    k0df_no_zeros = remove_cols_with_zeros(k0df_shared)
    metdf_no_zeros = remove_cols_with_zeros(metdf_shared)

    def demean_cols(M): return M - M.mean(axis=0)
    def fractionise_rows(M): return (1 / M.sum(axis=1)).reshape(-1,1) * M

    if (prep_type[:2] == 'v1'):
        def preprocess_data(df):
            return demean_cols(np.log(fractionise_rows(df.to_numpy())))
        X,Y = list(map(preprocess_data,[k0df_no_zeros,metdf_no_zeros]))
        X_cols = k0df_no_zeros.columns
        Y_cols = metdf_no_zeros.columns
        index = k0df_no_zeros.index
        assert (index == metdf_no_zeros.index).all(), 'X,Y should have same index by construction'
        if (prep_type == 'v1') or (prep_type == 'v1_Kbigger'):
            # print statement useful if comparing different prep-types; otherwise distracting
            # v1 will be used by default, so won't stress the output dimensions
            #print(f'Shapes for X,Y are {X.shape, Y.shape} respectively')
            return X,Y,X_cols,Y_cols,index
        else:
            print(f'Shapes for full X,Y are {X.shape, Y.shape} respectively')
            def load_diagnoses():
                df = pd.read_csv('../real_data/hmp2_metadata.csv',usecols=['External ID','diagnosis'])
                subdf = (df.drop_duplicates()
                         .set_index('External ID')
                         .loc[index]
                        )
                msg = 'all rows should have clinical annotations'
                assert subdf.shape[0] == index.shape[0], msg
                return subdf
            subdf = load_diagnoses()
            if prep_type == 'v1UC':
                rows_dgns = (subdf['diagnosis'] == 'UC')
            elif prep_type == 'v1CD':
                rows_dgns = (subdf['diagnosis'] == 'CD')
            elif prep_type == 'v1nonIBD':
                rows_dgns = (subdf['diagnosis'] == 'nonIBD')
            else:
                raise Exception(f'prep_type {prep_type} unrecognised - typo?')
            print(f'Shapes for  returned X,Y are {X[rows_dgns].shape, Y[rows_dgns].shape} respectively')
            return X[rows_dgns], Y[rows_dgns], X_cols, Y_cols, index[rows_dgns]

    else:
        raise Exception(f'unrecognised prep_type {prep_type}- perhaps not yet implemented?')
