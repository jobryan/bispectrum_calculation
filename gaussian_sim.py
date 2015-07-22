'''
gaussian_sim.py

Created on July 22, 2014
Updated on July 22, 2014

@author:    Jon O'Bryan

@contact:   jobryan@uci.edu

@summary:   Calculate Gaussian simulations of Planck map
            
            (1) Load cltt (already masked and corrected from cltt.py)
            (2) 

@inputs:    Load maps and masks from downloaded files 
            (located in "data/CompMap_CMB-smica_2048_R1.11.fits" and 
            "data/CompMap_Mask_2048_R1.00.fits" respectively) and Mll (located 
            in "data/na_mll_ell_xxxx.npy", where "xxxx" is the number of ell modes
            used in the mll calculation) from a pre-computed file

            na_map: Downloaded from Planck data store, 
                    http://irsa.ipac.caltech.edu/data/Planck/release_1/...
            na_mask: Similar to na_map
            na_mll: Calculated using mll.py (in misc folder)

@outputs:   Power spectrum from masked Planck data then corrected using mode 
            coupling matrix

            na_cltt

            saved to output/na_cl21_data_g_sim_[i_sim].dat

@command:   mpirun -np 22 python gaussian_sim.py

'''

# Python imports
import time
import pickle
import itertools as it

# 3rd party imports
import numpy as np
from matplotlib import pyplot as plt
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

    return i_lmax, i_nside, s_fn_map, s_map_name, s_fn_mask, s_fn_mll

'''
Main: Default run
'''

def main():

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
    (i_lmax, i_nside, s_fn_map, s_map_name, 
        s_fn_mask, s_fn_mll) = get_params(s_fn_params)

    i_nsims = 100
    i_sim_start = 1000

    if (i_rank == 0):
        print ""
        print "Run parameters:"
        print "lmax: %i, nside: %i, map name: %s, nsims: %i" % (i_lmax, i_nside, 
            s_map_name, i_nsims)

    # Load cltt, mask, mll

    if (i_rank == 0):
        print ""
        print "Loading cltt (previously masked and corrected)..."

    s_fn_cltt = 'output/na_cltt.npy'
    na_cltt = np.load(s_fn_cltt)

    na_mask = hp.read_map(s_fn_mask)
    na_mll = np.load(s_fn_mll)
    na_mll_inv = np.linalg.inv(na_mll)

    # Create map and cltt sims

    if (i_rank == 0):
        print ""
        print "Creating map and cltt simulations..."
        print ""

    f_t1 = time.time()

    #for i_sim in range(i_rank, i_nsims, i_size):
    for i_sim in range(i_sim_start, i_nsims+i_sim_start):

        # Create gaussian map and then find power spectrum
        # na_map_sim = hp.synfast(na_cltt, i_nside, fwhm=0.00145444104333, 
        #     verbose=False)
        na_map_sim = hp.synfast(na_cltt, nside=i_nside)
        # verbose doesn't work on cirrus...
        #na_map_sim = hp.synfast(na_cltt, nside=i_nside, verbose=False)

        na_map_sim = na_map_sim * na_mask

        na_cltt_sim = hp.anafast(na_map_sim, lmax=i_lmax-1)
        na_cltt_sim = np.dot(na_mll_inv, na_cltt_sim)

        # Save sim map and sim cltt

        s_fn_map_sim = '/data-1/jobryan/fnl_Planck/sims/na_map_sim_%i.npy' % i_sim
        np.save(s_fn_map_sim, na_map_sim)
        print "Saving map simulation to %s..." % s_fn_map_sim
        s_fn_cltt_sim = '/data-1/jobryan/fnl_Planck/sims/na_cltt_sim_%i.npy' % i_sim
        np.save(s_fn_cltt_sim, na_cltt_sim)
        print "Saving power spectrum simulation to %s..." % s_fn_cltt_sim
        print "(Elapsed time: %s)" % str(time.time() - f_t1)

    return

if __name__ == '__main__':
    main()