'''
cltt.py

Created on May 23, 2014
Updated on May 23, 2014

@author:    Jon O'Bryan

@contact:   jobryan@uci.edu

@summary:   Calculate power spectrum from data (Planck maps) in the following 
            way:
            
            (1) Load Planck map(s) (na_map)
            (2) Load Planck mask(s) (na_mask)
            (3) Apply mask to map (na_map * na_mask)
            (4) Perform spherical harmonic transform (anafast function from 
                healpy) to extract power spectrum (na_cltt)
            (5) Load mode coupling matrix (na_mll) (see Eqn. 90 (arXiv: 
                1004.1409v2, "CMB Constraints on Primordial NG...")
            (6) Correct for mask using inverse mode coupling matrix 
                (na_mll_inv = np.linalg.inv(na_mll)) as Eqn. 89 
                (clttp = np.dot(na_mll_inv, na_cltt))

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

            saved to output/na_cltt.npy

'''

# Python imports
import time
import pickle
import itertools as it

# 3rd party imports
import numpy as np
from matplotlib import pyplot as plt
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

    return i_lmax, i_nside, s_fn_map, s_map_name, s_fn_mask, s_fn_mll

'''
Plot map
'''

def plot_map(na_map, s_title=''):

    print ""
    print "Plotting %s map" % s_title

    plt.figure(figsize=(10,6))
    hp.mollview(na_map)
    plt.title(s_title)
    plt.show()

    return

'''
Plot power spectrum
'''

def plot_ps(lna_ell, lna_ps, ls_labels, s_ylabel, s_title='', s_fn_plot=''):

    print ""
    print "Plotting %s power spectrum" % s_title

    ls_lines = ["-","--","-.",":"]
    o_linecycler = it.cycle(ls_lines)

    plt.figure(figsize=(10,6))
    
    for i_ in range(len(lna_ell)):
        na_ell = lna_ell[i_]
        na_ps = lna_ps[i_]
        s_label = ls_labels[i_]

        s_linestyle = next(o_linecycler)

        if s_label == '':
            plt.plot(na_ell, na_ell*(na_ell+1.)*na_ps/2.0/np.pi , 
                linestyle=s_linestyle)
        else:
            plt.plot(na_ell, na_ell*(na_ell+1.)*na_ps/2.0/np.pi , 
                linestyle=s_linestyle, label=r"%s" % s_label)

    plt.xlabel(r'$\ell$', fontsize=20)
    plt.ylabel(r'$\ell(\ell+1)$ %s $/2\pi$' % s_ylabel, fontsize=20)

    if len(ls_labels) > 0:
        plt.legend()
    
    if s_label != '':
        plt.title(s_title)
    
    if s_fn_plot != '':
        plt.savefig(s_fn_plot)

    plt.show()

    return

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

    # Load Planck map and mask

    print ""
    print "Loading map and mask..."

    na_map = hp.read_map(s_fn_map) # for Planck SMICA, units of K
    na_map = na_map / 1e6 / 2.7 # convert units to mK -> unitless
    na_map = hp.remove_dipole(na_map) # removes the dipole and monopole -- turn off other lines doing this...
    na_mask = hp.read_map(s_fn_mask)
    na_map_masked = na_map * na_mask
    na_alm = hp.map2alm(na_map_masked, lmax=i_lmax-1)
    s_fn_alm = 'output/na_alm_data.fits'
    hp.write_alm(s_fn_alm, na_alm)

    # Spherical harmonic transform (map -> power spectrum)

    print ""
    print "Calculating power spectra..."

    na_cltt = hp.anafast(na_map_masked, lmax=i_lmax-1)
    na_wll = hp.anafast(na_mask, lmax=i_lmax-1)
    na_wll = na_wll + 2.0 # remove monopole and dipole
    na_ell = np.arange(len(na_cltt))
    na_ell = na_ell + 2.0 # remove monopole and dipole

    # Load mode coupling matrix and invert it

    print ""
    print "Loading and inverting mode coupling matrix..."

    na_mll = np.load(s_fn_mll)
    na_mll_inv = np.linalg.inv(na_mll)

    # Calculate Mll corrected power spectrum
    
    na_clttp = np.dot(na_mll_inv, na_cltt)

    # Save Mll corrected power spectrum
    s_fn_clttp = 'output/na_cltt.npy'
    np.save(s_fn_clttp, na_clttp)

    s_fn_cltt = 'output/na_cltt_not_corrected.npy'
    np.save(s_fn_cltt, na_cltt)

    print ""
    print "Saving power spectrum to %s" % s_fn_clttp

    '''
    Associated plots: map; mask; masked map; power spectrum of mask, power 
        spectrum of map, masked map, and mode coupling corrected map
    '''

    # NOTE: Mollview doesn't seem to work on cirrus -- probably an error with 
    #       this version of Healpy
    # plot_map(na_map, s_title='Raw Planck')
    # plot_map(na_mask, s_title='Mask')
    # plot_map(na_map_masked, s_title='Masked Map')

    plot_ps([na_ell], [na_wll], [''], s_ylabel='$W_\ell$', s_title='', 
        s_fn_plot='plots/fig_mask_ps.png')
    plot_ps([na_ell, na_ell], [na_cltt, na_clttp], 
        ['Masked', 'Masked, Corrected'], s_ylabel='$C_\ell$', s_title='', 
        s_fn_plot='plots/fig_masked_masked_corrected_ps2.png')

    return

if __name__ == '__main__':
    main()