'''
short_bi_ana.py

Created on May 19, 2014
Updated on May 23, 2014

@author:    Jon O'Bryan

@contact:   jobryan@uci.edu

@summary:   SHORTENED FORM OF bi_ana.py WITH JUST MPI BISPECTRUM CALCULATION:
            Calculate reduced bispectrum (analytical calculation) as given in 
            Eqn. 48 (arXiv: 1004.1409v2, "CMB Constraints on Primordial NG...")

@inputs:    Load alpha and beta arrays from a pre-computed file (currently, 
            "l_r_alpha_beta.txt")

            na_alpha: Calculated by compute_alphabeta.f90 in 
                /fnl_Planck/alphabeta_mod, following Eqn. 49
            na_beta: Similar to na_alpha

@outputs:   Analytical reduced bispectrum (see above)

            na_bi_ana

            saved to output/na_bi_ana.npy

@command:   To run for a given number of ell steps, e.g., 100 steps,

            python bi_ana.py 100

            The default number of steps is 80 which will occur upon running

            python bi_ana.py

            Currently set to not truncate the r steps; can be turned on for 
            linearly shorter run times (e.g., i_num_r_trunc = 40, etc.).

'''

# Python imports
import time
import itertools as it
import sys

# 3rd party imports
import numpy as np
from matplotlib import pyplot as plt
from mpi4py import MPI

'''
Load arrays
'''

def load_arr(i_col, s_fn, na_shape, b_1d=False):

    na_arr = np.loadtxt(s_fn, usecols=(i_col,), unpack=True, skiprows=3)
    
    if b_1d:
        na_arr = na_arr[:na_shape[0]]
    else:
        na_arr = na_arr.reshape(na_shape)    

    return na_arr

'''
Find closest set of points in source array to points in target array
'''

def closest_points(na_source, na_target, b_return_ind=False):

    na_ind = np.array([min(range(len(na_source)), 
                        key=lambda i: abs(na_source[i] - f_val)) 
                        for f_val in na_target])

    na_results = na_source[na_ind]

    if b_return_ind:
        return na_results, na_ind
    else:
        return na_results

'''
Select subarrays
'''

def sub_arr(na_arr, i_num_pts, f_max, s_spacing='lin'):

    # Select sub array of array using number of points, max, and spacing type

    if s_spacing == 'log':
        na_target_spacing = np.round(np.logspace(np.log10(np.min(na_arr)), 
                                        np.log10(f_max), i_num_pts))
    elif s_spacing == 'lin':
        na_target_spacing =  np.round(np.linspace(np.min(na_arr), f_max, 
                                        i_num_pts))

    na_arr_sub, na_ind = closest_points(na_arr, na_target_spacing, 
                                        b_return_ind=True)

    return na_arr_sub, na_ind

'''
Main: Default run
'''

def main(i_ell_trunc=9999):

    # MPI Initialization
    o_comm = MPI.COMM_WORLD
    i_rank = o_comm.Get_rank() # current core number -- e.g., i in arange(i_size)
    i_size = o_comm.Get_size() # number of cores assigned to run this program

    '''
    Load and plot data (ell, r, dr, alpha, beta)
    '''

    # Load parameters
    if i_rank == 0:
        print ""
        print "Setting load parameters:"

    s_fn_alpha_beta = 'data/l_r_alpha_beta.txt'
    i_num_r = 439
    i_num_ell = 1499

    na_shape = np.array([i_num_ell, i_num_r])

    if i_rank == 0:
        print "loading from:", s_fn_alpha_beta
        print "r steps:", i_num_r, "ell steps:", i_num_ell
        print ""

    # Load alpha, beta data
    if i_rank == 0:
        print "Loading data..."
        print "(displaying array shapes)"

    na_alpha = load_arr(3, s_fn_alpha_beta, na_shape)
    na_beta = load_arr(4, s_fn_alpha_beta, na_shape)
    na_l = np.arange(i_num_ell) + 2.
    na_r = load_arr(1, s_fn_alpha_beta, np.array([i_num_r]), b_1d=True)
    na_dr = load_arr(2, s_fn_alpha_beta, np.array([i_num_r]), b_1d=True)

    # Load power spectrum
    s_fn_clttp = 'output/na_cltt.npy'
    na_cltt = np.load(s_fn_clttp)


    if i_rank == 0:
        print ("alpha:", np.shape(na_alpha), "beta:", np.shape(na_beta), 
               "cltt:", np.shape(na_cltt))
        print "l:", np.shape(na_l), "r:", np.shape(na_r), "dr:", np.shape(na_dr)
        print ""


    '''
    Calculate bispectrum -- MPI optimized; only runs for more than one core
    '''

    # Run Parameters
    if i_rank == 0:
        print "(Running calculation with %i cores)" % i_size
        print "Setting run parameters for bispectrum calculation:"
    
    if i_ell_trunc != 9999:
        i_num_ell_trunc = i_ell_trunc
    else:
        i_num_ell_trunc = 80
    #i_num_r_trunc = 20
    i_num_r_trunc = len(na_r)

    if i_rank == 0:
        print "r steps (trunc):", i_num_r_trunc, "ell steps (trunc):", i_num_ell_trunc
        print ""

    # Chop down arrays for reduced bispectrum calculation
    if i_rank == 0:
        print "Truncating arrays for bispectrum calculation:"
        print "(displaying array shapes)"

    na_l, na_l_ind = sub_arr(na_l, i_num_ell_trunc, np.max(na_l), 
        s_spacing='lin')
    na_r, na_r_ind = sub_arr(na_r, i_num_r_trunc, np.max(na_r), 
        s_spacing='lin')

    na_dr = na_dr[na_r_ind]

    na_alpha_tmp = na_alpha[:, na_r_ind]
    na_a = na_alpha_tmp[na_l_ind,:]

    na_beta_tmp = na_beta[:, na_r_ind]
    na_b = na_beta_tmp[na_l_ind,:]

    if i_rank == 0:
        print "alpha:", np.shape(na_alpha), "beta:", np.shape(na_beta)
        print "l:", np.shape(na_l), "r:", np.shape(na_r), "dr:", np.shape(na_dr)
        print ""

    # Calculate bispectrum
    if i_rank == 0:
        print "Calculating analytical reduced bispectrum..."

    f_fnl = 1.0
    
    if i_rank == 0:
        f_tstart = time.time()
    
    '''
    Calculate bispectrum
    '''

    # b[l1,l2,l3] = 2 fnl int[ dr r^2 [a(r,l1) b(r,l2) b(r,l3) + cyc. perm] ]

    i_num_l = len(na_l)
    i_num_r = len(na_r)

    if i_rank == 0:
        na_bi_ana = np.zeros((i_num_l, i_num_l, i_num_l))

    for i_l1ind in range(i_num_l):

        if i_rank == 0:
            print "calculating ell_1 = %i" % na_l[i_l1ind]

        for i_l2ind in range(i_num_l):
            for i_l3ind in range(i_num_l):
                
                f_r_partial_sum = 0.0

                for i_r in range(i_rank, i_num_r, i_size):
                    f_r_partial_sum += (2.0 * f_fnl * 
                        na_dr[i_r] * na_r[i_r]**2.0 * 
            (na_a[i_l1ind, i_r] * na_b[i_l2ind, i_r] * na_b[i_l3ind, i_r]
            + na_a[i_l3ind, i_r] * na_b[i_l1ind, i_r] * na_b[i_l2ind, i_r]
            + na_a[i_l2ind, i_r] * na_b[i_l3ind, i_r] * na_b[i_l1ind, i_r]))

                f_r_sum = np.array(0., dtype='d') # recvbuffer for Reduce

                o_comm.Barrier()
                o_comm.Reduce(
                        [np.array(f_r_partial_sum, dtype='d'), MPI.DOUBLE], 
                        [f_r_sum, MPI.DOUBLE], op=MPI.SUM)

                if i_rank == 0:
                    na_bi_ana[i_l1ind, i_l2ind, i_l3ind] = f_r_sum
    
    if i_rank == 0:
        f_tstop = time.time()

    if i_rank == 0:
        print "Time for bispectrum calculation: %.2f" % (f_tstop - f_tstart)
        print ""

    # Save bispectrum as numpy array
    s_fn_bi = ('output/na_bi_ana_%i_rsteps_%i_ellsteps' 
                   % (i_num_r_trunc, i_num_ell_trunc))
    if i_rank == 0:
        print "Saving bispectrum to %s" % s_fn_bi
        np.save(s_fn_bi, na_bi_ana)

    if i_rank == 0:
        print "Done!"


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(int(sys.argv[1]))
    else:
        main()