'''
cl21_data_serial.py

Created on July 2, 2014
Updated on July 2, 2014

@author:    Jon O'Bryan

@contact:   jobryan@uci.edu

@summary:   cl21_data.py without MPI (for debugging)

@inputs:    Load cltt, alpha, beta, and beam from pre-computed files
            (located in "output/na_cltt.npy", "data/l_r_alpha_beta.txt", and 
            "output/na_bl.npy" respectively)

            na_cltt: Created by cltt.py
            na_alpha: Calculated by compute_alphabeta.f90 in 
                /fnl_Planck/alphabeta_mod, following Eqn. 49
            na_beta: Similar to na_alpha
            na_r: Similar to na_alpha
            na_dr: Similar to na_alpha
            na_ell: Similar to na_alpha
            na_bl: Calculated by calc_beam.py in misc/

@outputs:   Full skewness power spectrum from data,

            na_cl21_data

            saved to output/na_cl21_data.npy

@command:   ** Needs to run on elgordo due to strange MPI (slash mpi4py) issues
            on cirrus.

            mpirun -np 12 python -W ignore cl21_data.py

'''

# Python imports
import time
import pickle
import itertools as it

# 3rd party imports
import numpy as np
import healpy as hp

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
    s_fn_beam = d_params['s_fn_beam']
    s_fn_alphabeta = d_params['s_fn_alphabeta']
    s_fn_cltt = d_params['s_fn_cltt']

    return (i_lmax, i_nside, s_fn_map, s_map_name, s_fn_mask, s_fn_mll, 
        s_fn_beam, s_fn_alphabeta, s_fn_cltt)

'''
Main: Default run
'''

def main():

    '''
    Loading and calculating power spectrum components
    '''

    # Get run parameters
    
    s_fn_params = 'data/params.pkl'
    (i_lmax, i_nside, s_fn_map, s_map_name, s_fn_mask, s_fn_mll, s_fn_beam, 
        s_fn_alphabeta, s_fn_cltt) = get_params(s_fn_params)

    s_fn_cl21_data = 'output/na_cl21_data.dat'

    f_t1 = time.time()

    print ""
    print "Run parameters:"
    print "lmax: %i, nside: %i, map name: %s" % (i_lmax, i_nside, s_map_name)
    print "beam: %s, alpha_beta: %s, cltt: %s" % (s_fn_beam, s_fn_alphabeta, s_fn_cltt)

    print ""
    print "Loading ell, r, dr, alpha, beta, cltt, and beam..."

    na_l, na_r, na_dr, na_alpha, na_beta = np.loadtxt(s_fn_alphabeta, 
                                usecols=(0,1,2,3,4), unpack=True, skiprows=3)

    na_l = np.unique(na_l)
    na_r = np.unique(na_r)[::-1]
    na_l = na_l[:i_lmax]

    i_num_ell = len(na_l)
    i_num_r = len(na_r)

    print "i_num_r: %i, i_num_ell: %i" % (i_num_r, i_num_ell)

    na_alpha = na_alpha.reshape(i_num_ell, i_num_r)
    na_beta = na_beta.reshape(i_num_ell, i_num_r)
    na_dr = na_dr.reshape(i_num_ell, i_num_r)
    na_dr = na_dr[0]

    na_cltt = np.load(s_fn_cltt)
    na_cltt = na_cltt[:i_num_ell]

    na_bl = np.load(s_fn_beam)
    na_bl = na_bl[:i_num_ell]

    # f_t2 = time.time()

    print ""
    print "Calculating full skewness power spectrum..."

    na_alm = hp.synalm(na_cltt, lmax=i_num_ell, verbose=False)

    # f_t3 = time.time()

    na_cl21_data = np.zeros(i_num_ell)

    for i_r in range(i_num_r):

        if (i_r % (i_num_r / 10) == 0):
            print "Finished %i%% of jobs... (%.2f s)" % (i_r * 100 / i_num_r,
            time.time() - f_t1)


        na_Alm = hp.almxfl(na_alm, na_alpha[:,i_r] / na_cltt * na_bl)
        na_Blm = hp.almxfl(na_alm, na_beta[:,i_r] / na_cltt * na_bl)

        # f_t4 = time.time()

        na_An = hp.alm2map(na_Alm, nside=i_nside, fwhm=0.00145444104333, 
            verbose=False)
        na_Bn = hp.alm2map(na_Blm, nside=i_nside, fwhm=0.00145444104333, 
            verbose=False)

        # f_t5 = time.time()

        #print "starting map2alm for r = %i on core %i" % (i_r, i_rank)

        na_B2lm = hp.map2alm(na_Bn*na_Bn, lmax=i_num_ell)
        na_ABlm = hp.map2alm(na_An*na_Bn, lmax=i_num_ell)

        #print "finished map2alm for r = %i on core %i" % (i_r, i_rank)

        # f_t6 = time.time()

        na_clAB2 = hp.alm2cl(na_Alm, na_B2lm, lmax=i_num_ell)
        na_clABB = hp.alm2cl(na_ABlm, na_Blm, lmax=i_num_ell)

        na_clAB2 = na_clAB2[1:]
        na_clABB = na_clABB[1:]

        #f_t7 = time.time()

        na_cl21_data += (na_clAB2 + 2 * na_clABB) * na_r[i_r]**2. * na_dr[i_r]

    f_t8 = time.time()

    print ""
    print "Saving power spectrum to %s" % s_fn_cl21_data

    np.savetxt(s_fn_cl21_data, na_cl21_data)

        # print "Finished in %.2f s" % (f_t8 - f_t1)
        # # print "Load time: %.2f s" % (f_t2 - f_t1)
        # # print "synalm time: %.2f s" % (f_t3 - f_t2)
        # # print "almxfl time: %.2f s" % ((f_t4 - f_t3) / 2.)
        # # print "alm2map time: %.2f s" % ((f_t5 - f_t4) / 2.)
        # # print "map2alm time: %.2f s" % ((f_t6 - f_t5) / 2.)
        # # print "alm2cl time: %.2f s" % ((f_t7 - f_t6) / 2.)

    return

if __name__ == '__main__':
    main()