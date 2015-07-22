'''
cl21_fnl_sim.py

Created on January 24, 2015
Updated on January 24, 2015

@author:    Jon O'Bryan

@contact:   jobryan@uci.edu

@summary:   Calculate full skewness power spectrum from gaussian sims following 
            Eqns. 51-59 from arxiv:1004.1409 (Joseph's NG paper):
            
            (1) Load Planck power spectrum, alpha, beta, and beam (na_cltt, 
                na_alpha, na_beta, na_bl)
            (2) Create optimally weighted maps:
                (a) Convert cltt to alm (na_alm)
                (b) For each r value used in alpha, beta, multiply 
                    alpha / Cl * bl by alm (na_Almr, na_Blmr)
                (c) For each r value used in alpha, beta, convert Alm, Blm to 
                    maps (na_Arn, na_Brn)
            (3) Calculate two-one power spectra
                (a) For each r value used in alpha, beta, multiply Alm * B^2lm 
                    and convert to cl, and for each ell in cl, divide by (2l+1) 
                    (similarly for AB * B) (na_clAB2r, na_clABBr)
            (4) Calculate full skewness power spectrum
                (a) Sum (Cl^(AB,B) + 2 Cl^(A,B^2)) over all r values 
                    (na_cl21_data)

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

            na_cl21_sim

            saved to output/na_cl21_sim_[i_sim].npy

@command:   ** Needs to run on elgordo due to strange MPI (slash mpi4py) issues
            on cirrus.

            mpirun -np 12 python -W ignore cl21_data_g.py [i_sim]

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
Main: Default run
'''

def main(i_sim=1):

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

    #s_fn_cltt = ('data/fnl_sims/cl_fnl_0_sim_%04d.fits' % (i_sim,))

    f_fnl = 1.0

    if (i_rank == 0):

        s_fn_cl21_data = ('data/cl21_fnl_sims/cl21_fnl_%i_sim_%04d.dat' 
            % (int(f_fnl), i_sim))
        s_fn_cl21_data_no_mll = ('data/cl21_fnl_sims/cl21_fnl_%i_sim_%04d_no_mll.dat' 
            % (int(f_fnl), i_sim))

        f_t1 = time.time()

        print ""
        print "Run parameters:"
        print "(Using %i cores)" % i_size
        print "lmax: %i, nside: %i, map name: %s" % (i_lmax, i_nside, s_map_name)
        print "beam: %s, alpha_beta: %s, cltt: %s" % (s_fn_beam, s_fn_alphabeta, s_fn_cltt)

        print ""
        print "Loading ell, r, dr, alpha, beta, cltt, and beam..."

    i_lmax = 1024

    na_mask = hp.read_map(s_fn_mask)
    s_fn_mll = 'output/na_mll_%i_lmax.npy' % i_lmax
    na_mll = np.load(s_fn_mll)
    na_mll_inv = np.linalg.inv(na_mll)

    na_l, na_r, na_dr, na_alpha, na_beta = np.loadtxt(s_fn_alphabeta, 
                                usecols=(0,1,2,3,4), unpack=True, skiprows=3)

    na_l = np.unique(na_l)
    na_r = np.unique(na_r)[::-1]
    na_l = na_l[:i_lmax]

    i_num_ell = len(na_l)
    i_num_r = len(na_r)

    if (i_rank == 0):
        print "i_num_r: %i, i_num_ell: %i" % (i_num_r, i_num_ell)

    # assuming number of r's is set correctly and number of ells isn't necessarily
    # related to lmax, i_num_ell, etc.
    i_num_ell_alpha_beta_r = len(na_alpha)/i_num_r
    na_alpha = na_alpha.reshape(i_num_ell_alpha_beta_r, i_num_r)
    na_beta = na_beta.reshape(i_num_ell_alpha_beta_r, i_num_r)
    na_dr = na_dr.reshape(i_num_ell_alpha_beta_r, i_num_r)
    na_dr = na_dr[0]

    na_alpha = na_alpha[:i_num_ell,:]
    na_beta = na_beta[:i_num_ell,:]
    na_dr = na_dr[:i_num_ell]

    try:
        na_cltt = np.load(s_fn_cltt)
    except:
        na_cltt = np.loadtxt(s_fn_cltt)
    na_cltt = na_cltt[:i_num_ell]

    na_bl = np.load(s_fn_beam)
    na_bl = na_bl[:i_num_ell]

    # f_t2 = time.time()

    if (i_rank == 0):
        print ""
        print "Calculating full skewness power spectrum..."

    #na_alm = hp.synalm(na_cltt, lmax=i_num_ell, verbose=False)

    s_fn_almg = ('data/fnl_sims/alm_l_%04d_v3.fits' % (i_sim,))
    na_almg = hp.read_alm(s_fn_almg)
    na_almg = na_almg[:hp.Alm.getsize(i_num_ell)]
    s_fn_almng = ('data/fnl_sims/alm_nl_%04d_v3.fits' % (i_sim,))
    na_almng = hp.read_alm(s_fn_almng)
    na_almng = na_almng[:hp.Alm.getsize(i_num_ell)]
    na_alm = na_almg + f_fnl * na_almng

    # f_t3 = time.time()

    na_cl21_data = np.zeros(i_num_ell)
    na_work = np.zeros(1, dtype='i')
    na_result = np.zeros(i_num_ell, dtype='d')

    # if (i_rank == 0):
    #     print ("na_alm: %s, na_alpha: %s, na_beta: %s, na_cltt: %s, na_bl: %s" %
    #         (str(np.shape(na_alm)), str(np.shape(na_alpha)), 
    #             str(np.shape(na_beta)), str(np.shape(na_cltt)),
    #             str(np.shape(na_bl)) ))
    # else:
    #     print "core: %i" % i_rank
    #     print ("na_alm: %s, na_alpha: %s, na_beta: %s, na_cltt: %s, na_bl: %s" %
    #         (str(np.shape(na_alm)), str(np.shape(na_alpha)), 
    #             str(np.shape(na_beta)), str(np.shape(na_cltt)),
    #             str(np.shape(na_bl)) ))

    # master loop
    if (i_rank == 0):

        # send initial jobs

        # print ("na_alm: %s, na_alpha: %s, na_beta: %s, na_cltt: %s, na_bl: %s" %
        #         (str(np.shape(na_alm)), str(np.shape(na_alpha)), 
        #             str(np.shape(na_beta)), str(np.shape(na_cltt)),
        #             str(np.shape(na_bl)) ))

        for i_rank_out in range(1,i_size):

            na_work = np.array([i_rank_out-1], dtype='i')
            o_comm.Send([na_work, MPI.INT], dest=i_rank_out, tag=i_work_tag)

        for i_r in range(i_size-1,i_num_r):

            if (i_r % (i_num_r / 10) == 0):
                print "Finished %i%% of jobs... (%.2f s)" % (i_r * 100 / i_num_r,
                time.time() - f_t1)

            na_work = np.array([i_r], dtype='i')

            o_comm.Recv([na_result, MPI.DOUBLE], source=MPI.ANY_SOURCE, 
                status=o_status, tag=MPI.ANY_TAG)

            #print "received results from core %i" % o_status.Get_source()

            o_comm.Send([na_work,MPI.INT], dest=o_status.Get_source(), 
                tag=i_work_tag)

            na_cl21_data += na_result

        for i_rank_out in range(1,i_size):

            o_comm.Recv([na_result, MPI.DOUBLE], source=MPI.ANY_SOURCE,
                status=o_status, tag=MPI.ANY_TAG)

            na_cl21_data += na_result

            print "cl21 so far..."
            print na_cl21_data[:10], na_cl21_data[10:]

            o_comm.Send([np.array([9999], dtype='i'), MPI.INT], 
                dest=o_status.Get_source(), tag=i_die_tag)

    #slave loop:
    else:

        while(1):

            o_comm.Recv([na_work, MPI.INT], source=0, status=o_status, 
                tag=MPI.ANY_TAG)

            if (o_status.Get_tag() == i_die_tag):

                break

            i_r = na_work[0]

            #print "doing work for r = %i on core %i" % (i_r, i_rank)

            # print ("na_alm: %s, na_alpha: %s, na_beta: %s, na_cltt: %s, na_bl: %s" %
            #         (str(np.shape(na_alm)), str(np.shape(na_alpha)), 
            #             str(np.shape(na_beta)), str(np.shape(na_cltt)),
            #             str(np.shape(na_bl)) ))
            na_Alm = hp.almxfl(na_alm, np.nan_to_num(na_alpha[:,i_r] / na_cltt) * na_bl)
            na_Blm = hp.almxfl(na_alm, np.nan_to_num(na_beta[:,i_r] / na_cltt) * na_bl)

            # f_t4 = time.time()

            na_An = hp.alm2map(na_Alm, nside=i_nside, fwhm=0.00145444104333,
                verbose=False)
            na_Bn = hp.alm2map(na_Blm, nside=i_nside, fwhm=0.00145444104333,
                verbose=False)

            # *REMBER TO MULTIPLY BY THE MASK!*

            na_An = na_An * na_mask
            na_Bn = na_Bn * na_mask

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

            # print ("na_clAB2: %s, na_clABB: %s, na_r: %s, na_dr: %s" %
            #         (str(np.shape(na_clAB2)), str(np.shape(na_clABB)), 
            #             str(np.shape(na_r)), str(np.shape(na_dr)) ))

            na_result = np.zeros(i_num_ell, dtype='d')
            na_result += (na_clAB2 + 2 * na_clABB) * na_r[i_r]**2. * na_dr[i_r]

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
        print ""
        print ("Saving power spectrum to %s (not mll corrected)" 
            % s_fn_cl21_data_no_mll)

        np.savetxt(s_fn_cl21_data_no_mll, na_cl21_data)

        print ""
        print "Saving power spectrum to %s (mll corrected)" % s_fn_cl21_data
        
        na_cl21_data = np.dot(na_mll_inv, na_cl21_data)
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
    if (len(sys.argv) > 1):
        main(int(sys.argv[1]))
    else:
        main()