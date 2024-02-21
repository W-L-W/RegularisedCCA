import numpy as np
from typing import Iterable

from real_data.loading import get_dataset
from src.scaffold.core import Data, CV, MVNData, MVNCV

# COORDINATION
##############
def get_cv_object(dataset: str, algo: str) -> CV:
    """ Inputs are strings representing the dataset and algorithm respectively"""
    data = get_dataset(dataset)
    cv_obj = CV(data,folds=5,algo=algo,K=10)
    return cv_obj

def get_cv_obj_from_data(data: Data, algo: str) -> CV: #MVN=False
    """ For all simulations we use 5 folds and top 10 directions; this helper function abstracts that out """
    if type(data) == MVNData: # if dataset is generated from known multivariate normal distribution
        return MVNCV(data,folds=5,algo=algo,K=10)
    else:
        return CV(data,folds=5,algo=algo,K=10)
    
def compute_everything(cv_obj: CV, pens: Iterable[float]):
    cv_obj.fit_algo(pens)
    if type(cv_obj) == MVNCV:
        cv_obj.process_oracle()
    cv_obj.process_cv()
    cv_obj.process_stab()
    

