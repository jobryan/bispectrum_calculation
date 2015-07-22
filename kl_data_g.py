'''
kl_data_g.py

Created on July 22, 2014
Updated on July 22, 2014

@author:    Jon O'Bryan

@contact:   jobryan@uci.edu

@summary:   Calculate full kurtosis power spectra from gaussian sims following 
            Eqns. 68-74 from arxiv:1004.1409 (Joseph's NG paper):
            
            (1) Load Planck power spectrum, alpha, beta, and beam (na_cltt, 
                na_alpha, na_beta, na_bl)
            (2) Create optimally weighted maps:
                (a) Convert cltt to alm (na_alm)
                (b) For each r1, r2 value used in alpha, beta, multiply 
                    alpha / Cl * bl by alm (na_Almr1, na_Blmr1, na_Almr2, 
                    na_Blmr2)
                (c) For each r1, r2 value used in alpha, beta, convert Alm, Blm 
                    to maps (na_Ar1n, na_Br1n, na_Ar2n, na_Br2n)
            (3) Calculate two-two and three-one power spectra
                (a) For each r1, r2 value used in alpha, beta, multiply 
                    Alm * B^2lm and convert to cl, and for each ell in cl,
                    divide by (2l+1) (similarly for AB * B) 
                    (na_Jl_AB_ABr1, etc.)
            (4) Calculate full kurtosis power spectra
                (a) Sum over all r values (na_kl22_data, na_kl31_data)

@inputs:    Load cltt, alpha, beta, and beam from pre-computed files
            (located in "output/na_cltt.npy", "data/l_r_alpha_beta.txt", and 
            "output/na_bl.npy" respectively)

            na_cltt_sim: Created by gaussian_sim.py
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

            output/na_kl22_data_g_sim_[i_sim]_[i_num_r_run]_rsteps_[i_lmax_run]_lmax.dat
            output/na_kl31_data_g_sim_[i_sim]_[i_num_r_run]_rsteps_[i_lmax_run]_lmax.dat

@command:   ** Needs to run on elgordo due to strange MPI (slash mpi4py) issues
            on cirrus. Two paramters are i_lmax_run and i_num_r_run, 
            respectively.

            mpirun -np 12 python -W ignore kl_data_g.py [i_sim] 400 50

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
def main(i_sim=0):

    '''
    MPI Setup
    '''
    o_comm = MPI.COMM_WORLD
    i_rank = o_comm.Get_rank() # current core number -- e.g., i in arange(i_size)
    i_size = o_comm.Get_size() # number of cores assigned to run this program
    o_status = MPI.Status()

    i_work_tag = 0
    i_die_tag = 1

    '''
    Loading and calculating power spectrum components
    '''

    # Get run parameters
    
    s_fn_params = 'data/params.pkl'
    (i_lmax, i_nside, s_fn_map, s_map_name, s_fn_mask, s_fn_mll, s_fn_beam, 
        s_fn_alphabeta, s_fn_cltt) = get_params(s_fn_params)

    s_fn_cltt = ('sims/na_cltt_sim_%i.npy' % i_sim)

    if (i_rank == 0):

        f_t1 = time.time()

        print ""
        print "Run parameters:"
        print "(Using %i cores)" % i_size
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

    if (i_rank == 0):
        print "(sizes from file load)"
        print "i_num_r: %i, i_num_ell: %i" % (i_num_r, i_num_ell)

    if (len(sys.argv) > 2):
        i_lmax_run = int(sys.argv[2])
    else:
        i_lmax_run = i_lmax
    if (len(sys.argv) > 3):
        i_num_r_run = int(sys.argv[3])
    else:
        i_num_r_run = i_num_r

    i_lmax_run = min(i_lmax_run, len(na_l))
    i_num_r_run = min(i_num_r, i_num_r_run)

    i_r_steps = i_num_r / i_num_r_run

    na_mask = hp.read_map(s_fn_mask)
    s_fn_mll = 'output/na_mll_%i_lmax.npy' % i_lmax_run
    na_mll = np.load(s_fn_mll)
    na_mll_inv = np.linalg.inv(na_mll)

    if (i_rank == 0):
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

    if (i_rank == 0):
        print ""
        print "Calculating full kurtosis power spectra..."

    na_alm = hp.synalm(na_cltt, lmax=i_lmax_run, verbose=False)

    # f_t3 = time.time()

    na_work = np.zeros(2, dtype='i')
    na_result = np.zeros((2,i_lmax_run), dtype='d')
    li_dims = [i_num_r_run, i_num_r_run]

    # master loop
    if (i_rank == 0):

        na_kl22_data = np.zeros(i_lmax_run)
        na_kl31_data = np.zeros(i_lmax_run)

        # send initial jobs

        for i_rank_out in range(1, i_size):

            na_work = np.array(cart_index(i_rank_out-1, li_dims), dtype='i')
            o_comm.Send([na_work, MPI.INT], dest=i_rank_out, tag=i_work_tag)

        na_work = np.array(cart_index(i_size-1, li_dims), dtype='i')
        i_r1_start = na_work[0]
        i_r2_start = na_work[1]

        for i_r1 in range(i_r1_start, i_num_r_run):

            if (i_r1 % (i_num_r / 10) == 0):
                print "Finished %i%% of jobs... (%.2f s)" % (i_r1 * 100 / i_num_r_run,
                time.time() - f_t1)

            for i_r2 in range(i_r2_start, i_num_r_run):

                na_work = np.array([i_r1, i_r2], dtype='i')

                o_comm.Recv([na_result, MPI.DOUBLE], source=MPI.ANY_SOURCE, 
                    status=o_status, tag=MPI.ANY_TAG)

                #print "received results from core %i" % o_status.Get_source()

                o_comm.Send([na_work,MPI.INT], dest=o_status.Get_source(), 
                    tag=i_work_tag)

                na_kl22_data += na_result[0]
                na_kl31_data += na_result[1]

        for i_rank_out in range(1, i_size):

            o_comm.Recv([na_result, MPI.DOUBLE], source=MPI.ANY_SOURCE,
                status=o_status, tag=MPI.ANY_TAG)

            na_kl22_data += na_result[0]
            na_kl31_data += na_result[1]

            o_comm.Send([np.array([9999], dtype='i'), MPI.INT], 
                dest=o_status.Get_source(), tag=i_die_tag)

    #slave loop:
    else:

        while(1):

            o_comm.Recv([na_work, MPI.INT], source=0, status=o_status, 
                tag=MPI.ANY_TAG)

            if (o_status.Get_tag() == i_die_tag):

                break

            i_r1 = na_work[0]
            i_r2 = na_work[1]

            #print "doing work for r = %i on core %i" % (i_r, i_rank)

            na_Almr1 = hp.almxfl(na_alm, na_alpha[:,i_r1] / na_cltt * na_bl)
            na_Blmr1 = hp.almxfl(na_alm, na_beta[:,i_r1] / na_cltt * na_bl)
            na_Almr2 = hp.almxfl(na_alm, na_alpha[:,i_r2] / na_cltt * na_bl)
            na_Blmr2 = hp.almxfl(na_alm, na_beta[:,i_r2] / na_cltt * na_bl)

            # f_t4 = time.time() #all da maps

            na_Ar1n = hp.alm2map(na_Almr1, nside=i_nside, fwhm=0.00145444104333,
                verbose=False)
            na_Br1n = hp.alm2map(na_Blmr1, nside=i_nside, fwhm=0.00145444104333,
                verbose=False)
            na_Ar2n = hp.alm2map(na_Almr2, nside=i_nside, fwhm=0.00145444104333,
                verbose=False)
            na_Br2n = hp.alm2map(na_Blmr2, nside=i_nside, fwhm=0.00145444104333,
                verbose=False)

            na_Ar1n = na_Ar1n * na_mask
            na_Br1n = na_Br1n * na_mask
            na_Ar2n = na_Ar2n * na_mask
            na_Br2n = na_Br2n * na_mask

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

            na_result = np.zeros((2,i_lmax_run), dtype='d')
            if i_r1 == i_r2:
                na_result[0] += ((5./3.)**2. * na_Jl_AB_AB 
                * na_r[i_r1]**2. * na_dr[i_r1] * na_r[i_r2]**2. * na_dr[i_r2] 
                + 2. * na_Ll_AB_B2 * na_r[i_r1]**2. * na_dr[i_r1]) #kl22
                na_result[1] += ((5./3.)**2. * na_Jl_ABA_B 
                * na_r[i_r1]**2. * na_dr[i_r1] * na_r[i_r2]**2. * na_dr[i_r2] 
                + 2. * na_Ll_AB2_B * na_r[i_r1]**2. * na_dr[i_r1]) #kl31
            else:
                na_result[0] += ((5./3.)**2. * na_Jl_AB_AB 
                * na_r[i_r1]**2. * na_dr[i_r1] * na_r[i_r2]**2. * na_dr[i_r2]) #kl22
                na_result[1] += ((5./3.)**2. * na_Jl_ABA_B 
                * na_r[i_r1]**2. * na_dr[i_r1] * na_r[i_r2]**2. * na_dr[i_r2]) #kl31



            #print "finished work for r = %i on core %i" % (i_r, i_rank)

            o_comm.Send([na_result,MPI.DOUBLE], dest=0, tag=1)

            # print "Load time: %.2f s" % (f_t2 - f_t1)
            # print "synalm time: %.2f s" % (f_t3 - f_t2)
            # print "almxfl time: %.2f s" % ((f_t4 - f_t3) / 2.)
            # print "alm2map time: %.2f s" % ((f_t5 - f_t4) / 2.)
            # print "map2alm time: %.2f s" % ((f_t6 - f_t5) / 2.)
            # print "alm2cl time: %.2f s" % ((f_t7 - f_t6) / 2.)

    f_t8 = time.time()

    if (i_rank == 0):

        s_fn_kl22_data_no_mll = 'output/na_kl22_data_g_sim_%i_%i_rsteps_%i_lmax_no_mll.dat' % (i_sim, i_num_r_run, i_lmax_run)
        s_fn_kl31_data_no_mll = 'output/na_kl31_data_g_sim_%i_%i_rsteps_%i_lmax_no_mll.dat' % (i_sim, i_num_r_run, i_lmax_run)

        print ""
        print "Saving power spectrum to %s (not mll corrected)" % s_fn_kl22_data_no_mll
        print "Saving power spectrum to %s (not mll corrected)" % s_fn_kl31_data_no_mll

        np.savetxt(s_fn_kl22_data_no_mll, na_kl22_data)
        np.savetxt(s_fn_kl31_data_no_mll, na_kl31_data)

        s_fn_kl22_data = 'output/na_kl22_data_g_sim_%i_%i_rsteps_%i_lmax.dat' % (i_sim, i_num_r_run, i_lmax_run)
        s_fn_kl31_data = 'output/na_kl31_data_g_sim_%i_%i_rsteps_%i_lmax.dat' % (i_sim, i_num_r_run, i_lmax_run)

        print ""
        print "Saving power spectrum to %s" % s_fn_kl22_data
        print "Saving power spectrum to %s" % s_fn_kl31_data

        na_kl22_data = np.dot(na_mll_inv, na_kl22_data)
        na_kl31_data = np.dot(na_mll_inv, na_kl31_data)
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
    if (len(sys.argv) > 1):
        main(int(sys.argv[1]))
    else:
        main()