# to avoid pip installing package
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.utils.linalg import nsqrtm, sq_trigs

def test_nsqrtm():
    # toy 2 x 2 setting
    # initialise rotation matrix
    theta = np.pi/7
    O = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    # initialise positive semi-definite matrix
    e_vals = np.array([4,0])
    M = O @ np.diag(e_vals) @ O.T
    # check that the inverse square root is as expected
    sqrt = nsqrtm(M)
    assert np.allclose(sqrt, O @ np.diag([1/2, 0]) @ O.T)

def test_sq_trigs():
    # another toy setting where we can 'read off' the canonical angles between successive columns
    A = np.block([[np.eye(2)],[np.zeros((2,2))]])
    thetas = np.array([np.pi/7, np.pi/5])
    B = np.block([[np.diag(np.cos(thetas))], [np.diag(np.sin(thetas))]])

    out = sq_trigs(A,B, mode='cos')
    assert np.allclose(out, np.cumsum([np.cos(thetas)**2]))




