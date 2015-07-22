'''
cl21_ana.py

Created on May 23, 2014
Updated on May 23, 2014

@author:    Jon O'Bryan

@contact:   jobryan@uci.edu

@summary:   Calculate skewness power specrum estimator from analytical 
			bispectrum as given in Eqn. 60 (arXiv: 1004.1409v2, "CMB Constraints 
			on Primordial NG...")

@inputs:    Load bispectrum arrays from a pre-computed file (currently, 
            "output/ni_bi_ana.npy" -- very large file, around 64 GB)

            na_alpha: Calculated by compute_alphabeta.f90 in 
                /fnl_Planck/alphabeta_mod, following Eqn. 49
            na_beta: Similar to na_alpha

@outputs:   Analytical reduced bispectrum (see above)

            na_bi_ana

            saved to output/na_bi_ana.npy

'''

# Python imports
import time
import itertools as it

# 3rd party imports
import numpy as np
from matplotlib import pyplot as plt
from mpi4py import MPI

'''
Get parameters
'''

def get_params(s_fn):

    d_params = pickle.load(open(s_fn, 'rb'))
    
    i_lmax = d_params['i_lmax']
    i_nside = d_params['i_nside']
    s_fn_map = d_params['s_fn_map']
    s_map_name = d_params['s_map_name']
    s_fn_mask = d_params['s_fn_mask']
    s_fn_mll = d_params['s_fn_mll']

    return i_lmax, i_nside, s_fn_map, s_map_name, s_fn_mask, s_fn_mll

'''
Load power spectrum
'''


'''
Load analytical bispectrum
'''


'''
Sum over ells
'''


'''
Main: Default run
'''

def main():

    '''
    Loading and calculating power spectrum components
    '''

    # Get run parameters
    
    s_fn_params = 'data/params.pkl'
    (i_lmax, i_nside, s_fn_map, s_map_name, 
        s_fn_mask, s_fn_mll) = get_params(s_fn_params)

    print ""
    print "Run parameters:"

if __name__ == '__main__':
	main()