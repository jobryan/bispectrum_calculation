'''
fisher.py

Created on July 16, 2014
Updated on July 16, 2014

@author:    Jon O'Bryan

@contact:   jobryan@uci.edu

@summary:   Calculate Fisher bounds (e.g., Eqn. 80 from arxiv:1004.1409) and
            signal-to-noise for cl_21, kl_22, and kl_31.

@inputs:    Load cl_21, kl_22_gnl_0, kl_22_tnl_0, kl_31_gnl_0, kl_31_tnl_0 from
            pre-computed files (located in output folder, e.g., 
            "output/na_cl21_data.dat", etc.).

            na_cl21_data: output/na_cl21_data.dat 
            na_kl22_gnl_0_data: output/na_kl22_gnl_0_data.dat 
            na_kl22_tnl_0_data: output/na_kl22_tnl_0_data.dat 
            na_kl31_gnl_0_data: output/na_kl31_gnl_0_data.dat 
            na_kl31_tnl_0_data: output/na_kl31_tnl_0_data.dat 


@outputs:   Plots of the above saved to plots/[see above for names]

'''

# Python imports
import time
import pickle
import itertools as it

# 3rd party imports
import numpy as np


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
    print "lmax: %i, nside: %i, map name: %s" % (i_lmax, i_nside, s_map_name)

    # Filenames outside of standard run parameters

    s_fn_cl21_data = 'output/na_cl21_data.dat '
    s_fn_cl21_ana = 'output/cl_21_ana_437_rsteps_1299_ellsteps.dat'
    s_fn_kl22_data = 'output/na_kl22_data.dat'
    s_fn_kl22_ana = 'output/kl_22_ana_50_rsteps_100_Lsteps_400_lmax.dat'
    s_fn_kl31_data = 'output/na_kl31_data.dat'
    s_fn_kl31_ana = 'output/kl_31_ana_50_rsteps_100_Lsteps_400_lmax.dat'

    b_skewness = True
    b_kurtosis = False

    # Load skewness and kurtosis power spectra

    if b_skewness:

        print ""
        print "Loading skewness power spectrum..."

        na_cl21_data = np.loadtxt(s_fn_cl21_data)
        na_cl21_ana = np.loadtxt(s_fn_cl21_ana)

        na_l = np.arange(len(na_cl21_data)) + 2.0

        print ""
        print "Calculating signal-to-noise ratio for skewness..."

        f_sn_cl21 = np.sqrt(np.sum((2.0 * na_l + 1.0) * na_cl21_data))

        print "Signal-to-noise for skewness: %.3f" % f_sn_cl21

    if b_kurtosis:

        print ""
        print "Loading kurtosis power spectra..."
        
        na_kl22_data = np.loadtxt(s_fn_kl22_data)
        na_kl22_ana = np.loadtxt(s_fn_kl22_ana)
        na_kl31_data = np.loadtxt(s_fn_kl31_data)
        na_kl31_ana = np.loadtxt(s_fn_kl31_ana)

        na_l = np.arange(len(na_kl22_data)) + 2.0

        print ""
        print "Calculating signal-to-noise ratio for kl22 & kl31..."

        f_sn_kl22 = np.sqrt(np.sum((2.0 * na_l + 1.0) * na_kl22_data))
        f_sn_kl31 = np.sqrt(np.sum((2.0 * na_l + 1.0) * na_kl31_data))

        print "Signal-to-noise for kl22: %.3f" % f_sn_kl22
        print "Signal-to-noise for kl31: %.3f" % f_sn_kl31

        print ""
        print "Calculating Fisher bounds for tnl and gnl..."

        f_fisher_tnl = 

    '''
    Calculate Fisher 
    '''


    return

if __name__ == '__main__':
    main()