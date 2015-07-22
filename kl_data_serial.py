'''
kl_data_serial.py

Created on July 11, 2014
Updated on July 11, 2014

@author:    Jon O'Bryan

@contact:   jobryan@uci.edu

@summary:   kl_data.py without MPI (for debugging)

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

            na_kl22_data
            na_kl31_data

            saved to 

            output/na_kl22_data.npy
            output/na_kl31_data.npy

@command:   ** Needs to run on elgordo due to strange MPI (slash mpi4py) issues
            on cirrus. Two paramters are i_lmax_run and i_num_r_run, 
            respectively.

            mpirun -np 12 python -W ignore kl_data.py 400 50

'''

# Python imports
import time
import pickle
import itertools as it
import sys

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
Calculate Cartesian tuple from index
'''
def cart_index(i_index, li_dims):

    '''
    Returns the index from a cartesian product of vectors which each range from
    1 to n_i (where n_i is the ith entry of li_dims)
    '''
    i_dims = len(li_dims)
    na_tuple = np.zeros(i_dims)
    ''' Error checking '''
    if (i_index > np.prod(li_dims)):
        print "ERROR: Index too large for cart_index!"
        return na_tuple
    ''' Calculate value for each tuple entry '''
    for i in range(i_dims):
        if (i == i_dims - 1): 
            na_tuple[i] = i_index % li_dims[i]
        else:
            na_tuple[i] = i_index
            for j in range(i+1, i_dims):
                na_tuple[i] /= li_dims[j]
            na_tuple[i] %= li_dims[i]
    return na_tuple

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

    na_alpha = na_alpha.reshape(i_num_ell, i_num_r)
    na_beta = na_beta.reshape(i_num_ell, i_num_r)
    na_dr = na_dr.reshape(i_num_ell, i_num_r)
    na_dr = na_dr[0]

    print "(sizes from file load)"
    print "i_num_r: %i, i_num_ell: %i" % (i_num_r, i_num_ell)

    if (len(sys.argv) > 1):
        i_lmax_run = int(sys.argv[1])
    else:
        i_lmax_run = i_lmax
    if (len(sys.argv) > 2):
        i_num_r_run = int(sys.argv[2])
    else:
        i_num_r_run = i_num_r

    i_lmax_run = min(i_lmax_run, len(na_l))
    i_num_r_run = min(i_num_r, i_num_r_run)

    i_r_steps = i_num_r / i_num_r_run

    print "(sizes for run)"
    print "i_num_r_run: %i, i_lmax_run: %i" % (i_num_r_run, i_lmax_run)

    na_l = na_l[:i_lmax_run]
    na_r = na_r[::i_r_steps]
    na_dr = na_dr[::i_r_steps]

    na_alpha = na_alpha[:i_lmax_run, ::i_r_steps]
    na_beta = na_beta[:i_lmax_run, ::i_r_steps]

    na_cltt = np.load(s_fn_cltt)
    na_cltt = na_cltt[:i_lmax_run]

    na_bl = np.load(s_fn_beam)
    na_bl = na_bl[:i_lmax_run]

    # f_t2 = time.time()

    print ""
    print "Calculating full kurtosis power spectra..."

    na_alm = hp.synalm(na_cltt, lmax=i_lmax_run, verbose=False)

    # f_t3 = time.time()

    na_kl22_data = np.zeros(i_lmax_run)
    na_kl31_data = np.zeros(i_lmax_run)

    for i_r1 in range(i_num_r_run):

        if (i_r1 % (i_num_r / 10) == 0):
            print "Finished %i%% of jobs... (%.2f s)" % (i_r1 * 100 / i_num_r,
            time.time() - f_t1)

        for i_r2 in range(i_num_r_run):

            na_Almr1 = hp.almxfl(na_alm, na_alpha[:,i_r1] / na_cltt * na_bl)
            na_Blmr1 = hp.almxfl(na_alm, na_beta[:,i_r1] / na_cltt * na_bl)
            na_Almr2 = hp.almxfl(na_alm, na_alpha[:,i_r2] / na_cltt * na_bl)
            na_Blmr2 = hp.almxfl(na_alm, na_beta[:,i_r2] / na_cltt * na_bl)

            # f_t4 = time.time()

            na_Ar1n = hp.alm2map(na_Almr1, nside=i_nside, fwhm=0.00145444104333,
                verbose=False)
            na_Br1n = hp.alm2map(na_Blmr1, nside=i_nside, fwhm=0.00145444104333,
                verbose=False)
            na_Ar2n = hp.alm2map(na_Almr2, nside=i_nside, fwhm=0.00145444104333,
                verbose=False)
            na_Br2n = hp.alm2map(na_Blmr2, nside=i_nside, fwhm=0.00145444104333,
                verbose=False)

            # f_t5 = time.time()

            #print "starting map2alm for r = %i on core %i" % (i_r, i_rank)

            na_ABlmr1 = hp.map2alm(na_Ar1n*na_Br1n, lmax=i_lmax_run)
            if i_r1 == i_r2:
                na_B2lmr1 = hp.map2alm(na_Br1n*na_Br1n, lmax=i_lmax_run)
                na_AB2lmr1 = hp.map2alm(na_Ar1n*na_Br1n*na_Br1n, lmax=i_lmax_run)
            na_ABAlmr1 = hp.map2alm(na_Ar1n*na_Br1n*na_Ar1n, lmax=i_lmax_run)

            na_ABlmr2 = hp.map2alm(na_Ar2n*na_Br2n, lmax=i_lmax_run)
            na_B2lmr2 = hp.map2alm(na_Br2n*na_Br2n, lmax=i_lmax_run)            

            #print "finished map2alm for r = %i on core %i" % (i_r, i_rank)

            # f_t6 = time.time()

            na_Jl_ABA_B = hp.alm2cl(na_ABAlmr1, na_Blmr2, lmax=i_lmax_run)
            na_Jl_AB_AB = hp.alm2cl(na_ABlmr1, na_ABlmr2, lmax=i_lmax_run)

            na_Jl_ABA_B = na_Jl_ABA_B[1:]
            na_Jl_AB_AB = na_Jl_AB_AB[1:]

            if i_r1 == i_r2:

                na_Ll_AB2_B = hp.alm2cl(na_AB2lmr1, na_Blmr1, lmax=i_lmax_run)
                na_Ll_AB_B2 = hp.alm2cl(na_ABlmr1, na_B2lmr1, lmax=i_lmax_run)

                na_Ll_AB2_B = na_Ll_AB2_B[1:]
                na_Ll_AB_B2 = na_Ll_AB_B2[1:]

            #f_t7 = time.time()

            if i_r1 == i_r2:
                na_kl22_data += ((5./3.)**2. * na_Jl_AB_AB 
                * na_r[i_r1]**2. * na_dr[i_r1] * na_r[i_r2]**2. * na_dr[i_r2] 
                + 2. * na_Ll_AB_B2 * na_r[i_r1]**2. * na_dr[i_r1]) #kl22
                na_kl31_data += ((5./3.)**2. * na_Jl_ABA_B 
                * na_r[i_r1]**2. * na_dr[i_r1] * na_r[i_r2]**2. * na_dr[i_r2] 
                + 2. * na_Ll_AB2_B * na_r[i_r1]**2. * na_dr[i_r1]) #kl31
            else:
                na_kl22_data += ((5./3.)**2. * na_Jl_AB_AB 
                * na_r[i_r1]**2. * na_dr[i_r1] * na_r[i_r2]**2. * na_dr[i_r2]) #kl22
                na_kl31_data += ((5./3.)**2. * na_Jl_ABA_B 
                * na_r[i_r1]**2. * na_dr[i_r1] * na_r[i_r2]**2. * na_dr[i_r2]) #kl31

    f_t8 = time.time()

    s_fn_kl22_data = 'output/na_kl22_data_%i_rsteps_%i_lmax.dat' % (i_num_r_run, i_lmax_run)
    s_fn_kl31_data = 'output/na_kl31_data_%i_rsteps_%i_lmax.dat' % (i_num_r_run, i_lmax_run)

    print ""
    print "Saving power spectrum to %s" % s_fn_kl22_data
    print "Saving power spectrum to %s" % s_fn_kl31_data

    np.savetxt(s_fn_kl22_data, na_kl22_data)
    np.savetxt(s_fn_kl31_data, na_kl31_data)

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