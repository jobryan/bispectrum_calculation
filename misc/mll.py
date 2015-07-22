'''
mll.py

Created on May 23, 2014
Updated on May 23, 2014

@author:    Jon O'Bryan

@contact:   jobryan@uci.edu

@summary:   Calculate mode coupling matrix as given in Eqn. 90 (arXiv: 
            1004.1409v2, "CMB Constraints on Primordial NG..."), namely

            mll' = (2l'+1)/4pi sum[ (2l''+1) W_l'' wig3j(l,l',l'')**2, l'']

            where W_l'' is the power spectrum of the mask used.

@inputs:    Requires a function to calculate the Wigner 3j symbol (wigner.pyx)
            and a mask (located in "data/CompMap_Mask_2048_R1.00.fits").

            na_mask: Downloaded from Planck data store, 
                    http://irsa.ipac.caltech.edu/data/Planck/release_1/...

@outputs:   Mode coupling matrix for a given ell (square matrix of size 
            ell x ell)

            na_mll

            saved to output/na_mll_ell_xxxx.npy

            where "xxxx" is the given ell

'''

# Python imports
import time
import itertools as it

# 3rd party imports
import numpy as np
from matplotlib import pyplot as plt
import healpy as hp

# Local imports
from wigner import wigner

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
Calculate single mode coupling matrix entry
'''

def mll(i_l1, i_l2, na_wl, i_lmax):

    o_w = wigner()
    f_mll_entry = 0.0

    for i_l3 in range(i_lmax):
        
        if (abs(i_l1-i_l2) <= i_l3 
            and i_l3 <= abs(i_l1+i_l2) 
            and (i_l1+i_l2+i_l3)%2 == 0):
            
            f_mll_entry += ((2.0*i_l2+1.0)/(4.0*np.pi)*(2.0*i_l3+1.0)
                            *na_wl[i_l3]*o_w.w3j(i_l1,i_l2,i_l3)**2.0)

    return f_mll_entry

'''
Calculate entire mode coupling matrix
'''

def calc_mll(na_wl, i_lmax):
    
    na_mll = np.zeros((i_lmax, i_lmax))
    
    for i_row in range(i_lmax):

        for i_col in range(i_lmax):

            if (np.mod(i_row,100)==0 and np.mod(i_col,100)==0): 
                print "row, col:", i_row, i_col

            na_mll[i_row,i_col] = mll(i_row, i_col, na_wl, i_lmax)

    return na_mll

'''
Plot mode coupling matrix
'''

def plot_matrix(na_mat, ls_labels, s_fn_mat=''):

    plt.figure(figsize=(10,10))
    plt.imshow(na_mat, origin='lower')

    plt.colorbar()
    plt.xlabel(r"%s" % ls_labels[0], fontsize=20, weight='bold')
    plt.ylabel(r"%s" % ls_labels[1], fontsize=20, weight='bold')
    plt.title(r"%s" % ls_labels[2], fontsize=20, weight='bold')

    if s_fn_mat != '':
        plt.savefig(s_fn_mat)

    plt.show()

    return

'''
Main: Default run
'''

def main():

    # Get run parameters
    s_fn_params = 'data/params.pkl'
    (i_lmax, i_nside, s_fn_map, s_map_name, 
        s_fn_mask, s_fn_mll) = get_params(s_fn_params)

    # Load mask
    na_mask = hp.read_map(s_fn_mask)
    na_wl = hp.anafast(na_mask, lmax=i_lmax)

    # Calculate mode coupling matrix
    na_mll = calc_mll(na_wl, i_lmax)

    s_fn_mll = 'output/na_mll_ell_%i.npy' % i_lmax
    np.save(s_fn_mll, na_mll)

    # Plot mode coupling matrix
    ls_labels = ["$\ell$", "$\ell^'$", "$\log M_{\ell \ell^'}$"]
    s_fn_mat = 'plot/fig_mll.png'
    plot_matrix(np.log(na_mll), ls_labels, s_fn_mat)

    return

if __name__ == '__main__':
    main()