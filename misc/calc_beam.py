'''
calc_beam.py

Created on July 2, 2014
Updated on July 2, 2014

@author:    Jon O'Bryan

@contact:   jobryan@uci.edu

@summary:   Calculate Gaussian beam used in other calculations based off of FWHM
            for SMICA map using Healpy function gauss_beam()

@inputs:    FWHM in radians for Planck SMICA map (found in Planck 2013 Results.
            XII. Component separation, Table 1, arxiv: 1303.5072)

@outputs:   SMICA beam function

            na_bl

            saved to ../output/na_bl.npy

'''

# Python imports
import pickle

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

    # Get run parameters
    
    s_fn_params = '../data/params.pkl'
    (i_lmax, i_nside, s_fn_map, s_map_name, 
        s_fn_mask, s_fn_mll) = get_params(s_fn_params)

    f_fwhm_radians = 0.00145444

    na_bl = hp.gauss_beam(f_fwhm_radians, lmax) #only works on elgordo (hp 1.5+)
    na_ell = np.arange(len(na_bl))

    plot_ps([na_ell], [na_bl], ['SMICA'], s_ylabel='$b_\ell$', s_title='', 
        s_fn_plot='../plots/fig_beam_ps.png')

    # Save beam function
    na_bl = na_bl[2:] # remove monopole and dipole
    na_ell = na_ell[2:] # remove monopole and dipole

    s_fn_beam = '../output/na_bl.npy'
    np.save(s_fn_beam, na_bl)

    return

if __name__ == '__main__':
    main()