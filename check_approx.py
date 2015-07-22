'''
check_approx.py

Created on October 31, 2014
Updated on October 31, 2014

@author:    Jon O'Bryan

@contact:   jobryan@uci.edu

@summary:   Check the approximation in Regan, Eqn. 39 (arXiv:1310.8617v1) -- all
            other equations here reference arXiv:1004.1409v2.

@inputs:    Load cltt, alpha, beta, and beam from pre-computed files
            (located in "output/na_cltt.npy", "data/l_r_alpha_beta.txt", and 
            "output/na_bl.npy" respectively)

            na_cltt: Created by cltt.py
            na_alpha: Calculated by compute_alphabeta.f90 in 
                /fnl_Planck/alphabeta_mod, following Eqn. 49
            na_beta: Similar to na_alpha
            na_flr1r2: Calculated by compute_alphabeta_mod.f90 in 
                /fnl_Planck/alphabeta_mod, following Eqn. 67
            na_r: Similar to na_alpha
            na_dr: Similar to na_alpha
            na_ell: Similar to na_alpha
            na_clcurv: Calculated elsewhere (see other docs)

@outputs:   Results for small ell of full integration and approx,

            na_full_int
            na_approx

            saved to 

            output/na_full_int.dat
            output/na_approx.dat

@command:   ** Needs to run on elgordo due to strange MPI (slash mpi4py) issues
            on cirrus. Two paramters are i_lmax_run and i_num_r_run, 
            respectively.

            mpirun -np 12 python -W ignore check_approx.py

'''

# Python imports
import time
import pickle
import itertools as it
import sys

# 3rd party imports
import numpy as np
import healpy as hp
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
    s_fn_beam = d_params['s_fn_beam']
    s_fn_alphabeta = d_params['s_fn_alphabeta']
    s_fn_cltt = d_params['s_fn_cltt']
    return (i_lmax, i_nside, s_fn_map, s_map_name, s_fn_mask, s_fn_mll, 
        s_fn_beam, s_fn_alphabeta, s_fn_cltt)

'''
Full integral
'''
def full_int_sum(i_L, i_Lmax):
    f_sum = 0.0
    for l1 in range(i_Lmax):
        for l2 in range(i_Lmax):
            for l3 in range(i_Lmax):
                for l4 in range(i_Lmax):
                    for r1 in range(i_num_r):
                        for r2 in range(i_num_r):
                            f_sum += (na_dr[r1]*na_dr[r2]*
                                na_r1[r1]**2.*na_r2[r2]**2.*na_flr1r2[i_L,r1,r2]
                                *(na_alpha[l1,r1]*na_beta[l2,r1] + 
                                    na_alpha[l2,r1]*na_beta[l1,r1])
                                *(na_alpha[l3,r2]*na_beta[l4,r2] + 
                                    na_alpha[l4,r2]*na_beta[l3,r2]))
    return f_sum

'''
Approximation
'''
def approx_sum(i_L, i_Lmax):
    f_sum = 0.0
    for l1 in range(i_Lmax):
        for l2 in range(i_Lmax):
            for l3 in range(i_Lmax):
                for l4 in range(i_Lmax):
                    f_sum += (na_clcurv[i_L]*
                        (na_cltt[l1]+na_cltt[l2])*(na_cltt[l3]+na_cltt[l4]))
    return f_sum

'''
Main: Default run
'''
def main():

    # '''
    # MPI Setup
    # '''
    # o_comm = MPI.COMM_WORLD
    # i_rank = o_comm.Get_rank() # current core number -- e.g., i in arange(i_size)
    # i_size = o_comm.Get_size() # number of cores assigned to run this program
    # o_status = MPI.Status()

    # i_work_tag = 0
    # i_die_tag = 1

    '''
    Loading and calculating power spectrum components
    '''

    # Get run parameters
    
    s_fn_params = 'data/params.pkl'
    (i_lmax, i_nside, s_fn_map, s_map_name, s_fn_mask, s_fn_mll, s_fn_beam, 
        s_fn_alphabeta, s_fn_cltt) = get_params(s_fn_params)
    
    print ""
    print "Loading flr1r2..."

    s_fn_flr1r2 = 'data/l_r_r2_fr1r2_fixed.txt'

    f_t0 = time.time()

    na_l, na_r1, na_r2, na_dr, na_flr1r2 = np.loadtxt(s_fn_flr1r2,
        usecols=(0,1,2,3,4), unpack=True, skiprows=3)

    na_l = np.unique(na_l)
    na_r1 = np.unique(na_r1)[::-1]
    na_r2 = np.unique(na_r2)[::-1]
    na_l = na_l[:i_lmax]

    i_num_ell = len(na_l)
    i_num_r = len(na_r1)

    na_flr1r2 = na_alpha.reshape(i_num_ell, i_num_r, i_num_r)
    na_dr = na_dr.reshape(i_num_ell, i_num_r)
    na_dr = na_dr[0]

    f_t1 = time.time()

    print "Time to load flr1r2: %.2f" % (f_t1-f_t0)

    print ""
    print "Run parameters:"
    print "(Using %i cores)" % i_size
    print "lmax: %i, nside: %i, map name: %s" % (i_lmax, i_nside, s_map_name)
    print "beam: %s, alpha_beta: %s, cltt: %s" % (s_fn_beam, s_fn_alphabeta, s_fn_cltt)

    print ""
    print "Loading ell, r, dr, alpha, beta, and cltt..."

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

    na_cltt = np.load(s_fn_cltt)
    na_cltt = na_cltt[:i_lmax]

    print ""
    print "Loading clcurv..."

    s_fn_clcurv = 'data/l_r_clcurv.txt'

    na_clcurv = np.loadtxt(s_fn_clcurv, usecols=(2,), unpack=True, skiprows=2)

    # Calculate approximations

    i_Lmax = 20
    na_full_int = np.zeros(i_Lmax-2)
    na_approx = np.zeros(i_Lmax-2)

    for i_L in range(2, i_Lmax):
        na_full_int[i_L-2] = full_int_sum(i_L, i_Lmax)
        na_approx[i_L-2] = approx_sum(i_L, i_Lmax)

if __name__=='__main__':
    main()
# Load cltt, alpha, beta, flr1r2, r, dr, ell, cl_curv
