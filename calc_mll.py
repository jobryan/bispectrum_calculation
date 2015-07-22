'''
calc_mll.py

Created on March 22, 2013
Updated on September 3, 2014

@author:    Jon O'Bryan

@contact:   jobryan@uci.edu

@summary:   Calculate mode coupling matrix, Mll, as given in Eqn. 90 of 
            arxiv:1004.1409 (Joseph's NG paper), i.e.,



@inputs:    Load mask to be used (e.g., from Planck data) 

            na_mask: (from s_fn_mask)

@outputs:   Mode coupling matrix up to a certain ell value

            na_mll_[ell]_lmax.npy

            saved to output/na_mll_[ell]_lmax.npy

@command:   mpirun -np 12 python calc_mll.py [i_lmax]

'''

# Python imports
import time
import pickle
import sys

# 3rd party imports
import numpy as np
import healpy as hp
from mpi4py import MPI
from matplotlib import pyplot as plt

# Homemade imports
from wigner import wigner

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

def mll(i_l1, i_l2, na_wl, i_lmax):
    w = wigner()
    na_mll_tmp = 0.0

    # Summation -- reasonable loop here

    for i_l3 in range(i_lmax):
        if (abs(i_l1-i_l2) <= i_l3 and 
            i_l3 <= abs(i_l1+i_l2) and 
            (i_l1+i_l2+i_l3)%2 == 0):
            na_mll_tmp += ((2.0*i_l2+1.0)/(4.0*np.pi)*(2.0*i_l3+1.0)*na_wl[i_l3]
                *w.w3j(i_l1,i_l2,i_l3)**2.0)

    return na_mll_tmp

def main(i_lmax=1499):
    
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
    Set run parameters
    '''

    s_fn_params = 'data/params.pkl'
    (i_lmax_default, i_nside, s_fn_map, s_map_name, s_fn_mask, s_fn_mll, 
        s_fn_beam, s_fn_alphabeta, s_fn_cltt) = get_params(s_fn_params)

    '''
    Load data
    '''

    na_mask = hp.read_map(s_fn_mask)
    na_wl = hp.anafast(na_mask,lmax=i_lmax-1)

    s_fn_mll = 'output/na_mll_%i_lmax.npy' % i_lmax

    na_mll = np.zeros((i_lmax,i_lmax))
    na_mll_split = np.zeros((i_lmax,i_lmax))

    '''
    Calculate mll matrix
    '''

    if (i_rank == 0):
        print "Calculating mll matrix..."
        t0 = time.time()

    for i_row in range(i_rank, i_lmax, i_size):
        for i_col in range(i_lmax):
            if (np.mod(i_row,100)==0 and np.mod(i_col,100)==0): 
                print "row, col:", i_row, i_col
            na_mll_split[i_row,i_col] = mll(i_row,i_col,na_wl,i_lmax) 

    o_comm.Barrier()
    o_comm.Reduce(
            [np.array(na_mll_split, dtype='d'), MPI.DOUBLE], 
            [na_mll, MPI.DOUBLE], op=MPI.SUM)#, root=0)

    if (i_rank == 0):
        t1 = time.time()
        print "Time to calculate:", t1-t0

        print "Saving mll matrix to %s..." % s_fn_mll
        np.save(s_fn_mll, na_mll)
        
        b_plot = False
        if b_plot:
            print "Plotting mll matrix..."
            plt.imshow(np.log(na_mll), origin='lower')

            plt.colorbar()
            plt.xlabel(r"$\ell$", fontsize=20, weight='bold')
            plt.ylabel(r"$\ell^'$", fontsize=20, weight='bold')
            plt.title(r"$\log M_{\ell \ell^'}$", fontsize=20, weight='bold')

            plt.show()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(int(sys.argv[1]))
    else:
        main()