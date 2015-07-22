'''
plot_misc.py

Created on July 11, 2014
Updated on July 11, 2014

@author:    Jon O'Bryan

@contact:   jobryan@uci.edu

@summary:   Plot various plots used in paper:

            (1a) ell vs. ell*(ell+1)*cltt [K^2]
            (1b) ell vs. ell*(ell+1)*clcurv [K^2] (fig_cltt_clcurv.png)
            (2) ell vs. bl (nofig_beam.png)
            (3) ell vs. ell*(ell+1)*wll (fig_wl.png)
            (4) ell vs. ell' vs. log(Mll') (fig_Mll.png)
            (5a) ell vs. ell*(ell+1)*cltt_masked [K^2]
            (5b) ell vs. ell*(ell+1)*cltt_unmasked [K^2] (nofig_cltt_mask_nomask.png)
            (6a) ell vs. ell*(ell+1)*cl21_data [K^2]
            (6b) ell vs. ell*(ell+1)*cl21_ana [K^2] (nofig_cl21.png)
            (7a) ell vs. ell*(ell+1)*kl22_data [K^2]
            (7b) ell vs. ell*(ell+1)*kl22_ana [K^2] (nofig_kl22.png)
            (8a) ell vs. ell*(ell+1)*kl31_data [K^2]
            (8b) ell vs. ell*(ell+1)*kl31_ana [K^2] (nofig_kl31.png)
            (9a) ell vs. ell*(ell+1)*cltt_Planck
            (9b) ell vs. ell*(ell+1)*cltt_sims (fig_cltt_sims.png)
            (10a) ell vs. ell*(ell+1)*beta/(2*pi)/(10**-10)
            (10b) ell vs. alpha/(10**-10) [Mpc^-3] (fig1_alpha_beta_vs_ell.png)

@inputs:    Load maps and masks from downloaded files 
            (located in "data/CompMap_CMB-smica_2048_R1.11.fits" and 
            "data/CompMap_Mask_2048_R1.00.fits" respectively) and Mll 
            (located in "data/na_mll_ell_xxxx.npy", where "xxxx" is the number 
            of ell modes used in the mll calculation) from a pre-computed file.

            Load cl21_data, cl21_ana, kl22_data, kl22_ana, kl31_data, and 
            kl31_ana from pre-computed files (located in output folder, e.g., 
            "output/cl_21_ana_437_rsteps_1299_ellsteps.dat", etc.).

            na_map: Downloaded from Planck data store, 
                    http://irsa.ipac.caltech.edu/data/Planck/release_1/...
            na_mask: Similar to na_map
            na_mll: Calculated using mll.py (in misc folder)

            na_clcurv: output/na_clcurv_ell_1499.txt

            na_cl21_data: output/na_cl21_data.dat 
            na_cl21_ana: output/cl_21_ana_437_rsteps_1299_ellsteps.dat 
            na_kl22_data: output/na_kl22_data.dat 
            na_kl22_ana: output/kl_22_ana_50_rsteps_100_Lsteps_400_lmax.dat
            na_kl31_data: output/na_kl31_data.dat 
            na_kl31_ana: output/kl_31_ana_50_rsteps_100_Lsteps_400_lmax.dat

@outputs:   Plots of the above saved to plots/[see above for names]

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
Plot heatmap
'''

def plot_heatmap(na_data, s_xlabel='', s_ylabel='', s_title='', s_fn_plot='',
    b_log=False):

    print ""
    print "Plotting %s heatmap" % s_title

    if (b_log):
        na_data = np.log10(na_data)

    plt.figure(figsize=(10,6))
    plt.imshow(na_data, origin='lower right')
    plt.colorbar()

    if s_xlabel != '':
        plt.xlabel(r'%s' % s_xlabel, fontsize=20)
    if s_ylabel != '':
        plt.ylabel(r'%s' % s_ylabel, fontsize=20)
    if s_title != '':
        plt.title(r'%s' % s_title, fontsize=20)

    if s_fn_plot != '':
        plt.savefig(s_fn_plot)

    plt.show()

'''
Plot power spectrum
'''

def plot_ps(lna_ell, lna_ps, ls_labels, s_ylabel, s_title='', s_fn_plot='', 
    b_units_on=True):

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
    if b_units_on:
        plt.ylabel(r'$\ell(\ell+1)$ %s $/2\pi$ [$K^2$]' % s_ylabel, fontsize=20)
    else:
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
Plot function
'''

def plot_fn(lna_x, lna_y, ls_labels, s_xlabel, s_ylabel, s_title='', 
    s_fn_plot=''):

    print ""
    print "Plotting %s function" % s_title

    ls_lines = ["-","--","-.",":"]
    o_linecycler = it.cycle(ls_lines)

    plt.figure(figsize=(10,6))
    
    for i_ in range(len(lna_x)):
        na_x = lna_x[i_]
        na_y = lna_y[i_]
        s_label = ls_labels[i_]

        s_linestyle = next(o_linecycler)

        if s_label == '':
            plt.plot(na_x, na_y, linestyle=s_linestyle)
        else:
            plt.plot(na_x, na_y, linestyle=s_linestyle, label=r"%s" % s_label)

    plt.xlabel(r'%s' % s_xlabel, fontsize=20)
    plt.ylabel(r'%s' % s_ylabel, fontsize=20)

    if len(ls_labels) > 0:
        plt.legend()
    
    if s_label != '':
        plt.title(s_title)
    
    if s_fn_plot != '':
        plt.savefig(s_fn_plot)

    plt.show()

    return

'''
Plot alpha, veta vs. ell for various r values
'''

def plot_alpha_beta(na_a, na_b, na_r, na_r_plot, na_l, na_cltt, 
                    f_norm, s_fn=''):

    plt.figure(figsize=(12,8))
    ls_lines = ["-","--","-.",":"]
    o_linecycler = it.cycle(ls_lines)

    for f_r in na_r_plot:
        i_r_idx = np.where(na_r==f_r)[0][0]
        s_linestyle = next(o_linecycler)
        plt.subplot(211)
        plt.plot(na_l, na_l * (na_l + 1.0) * na_b[:, i_r_idx] / 2.0 / np.pi 
            / f_norm, linestyle=s_linestyle, 
            label=r"%.1f $\tau_*$" % ( (f_r - 14297.6746786552) / -235. ))
        plt.subplot(212)
        plt.plot(na_l, na_a[:, i_r_idx] / f_norm, linestyle=s_linestyle)
    
    plt.subplot(211)
    s_linestyle = next(o_linecycler)
    plt.plot(na_l, na_l * (na_l + 1.0) * na_cltt[2:len(na_l)+2] / 2.0 / np.pi 
             / 1000., linestyle=s_linestyle, label=r"$C_\ell^{\theta\theta}$")
    plt.ylabel(r"$\ell(\ell+1)\beta_\ell(r)/2\pi$ $[10^{-10}]$", 
        fontsize=20)
    plt.xscale('log')
    plt.xlim([np.min(na_l), np.max(na_l)])
    plt.legend(loc='upper left').draw_frame(False)
    plt.subplot(212)
    plt.ylabel(r"$\alpha_\ell(r)$ $[10^{-10}$ Mpc$^{-3}]$", 
        fontsize=20)
    plt.xscale('log')
    plt.xlim([np.min(na_l), np.max(na_l)])

    plt.xlabel(r"$\ell$", fontsize=20, weight='bold')

    if s_fn != '':
        plt.savefig(s_fn)

    plt.show()

    return

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

    # Filenames outside of standard run parameters

    s_fn_beam = 'output/na_bl.npy'
    s_fn_clcurv = 'output/na_clcurv_ell_1499.txt'
    s_fn_cl21_data = 'output/na_cl21_data.dat '
    s_fn_cl21_ana = 'output/cl_21_ana_437_rsteps_1299_ellsteps.dat'
    s_fn_kl22_data = 'output/na_kl22_data.dat'
    s_fn_kl22_ana = 'output/kl_22_ana_50_rsteps_100_Lsteps_400_lmax.dat'
    s_fn_kl31_data = 'output/na_kl31_data.dat'
    s_fn_kl31_ana = 'output/kl_31_ana_50_rsteps_100_Lsteps_400_lmax.dat'
    s_fn_cltt_planck = 'output/na_cltt.npy'
    s_fn_alphabeta = 'data/l_r_alpha_beta.txt'

    b_plot_main = False
    b_plot_skewness = False
    b_plot_kurtosis = False
    b_plot_sims = False
    b_plot_alpha_beta = True
    b_plot_misc = False

    # Load Planck map and mask

    print ""
    print "Loading map and mask..."

    na_map = hp.read_map(s_fn_map) # for Planck SMICA, units of uK
    na_map = na_map / 1e6 # convert units to K
    na_mask = hp.read_map(s_fn_mask)
    na_map_masked = na_map * na_mask

    # Spherical harmonic transform (map -> power spectrum)

    print ""
    print "Calculating power spectra..."

    na_cltt = hp.anafast(na_map_masked, lmax=i_lmax-1)
    na_wll = hp.anafast(na_mask, lmax=i_lmax-1)
    na_wll = na_wll[2:] # remove monopole and dipole
    na_ell = np.arange(len(na_cltt))
    na_ell = na_ell[2:] # remove monopole and dipole

    # Load mode coupling matrix and invert it

    print ""
    print "Loading and inverting mode coupling matrix..."

    na_mll = np.load(s_fn_mll)
    na_mll_inv = np.linalg.inv(na_mll)

    # Calculate Mll corrected power spectrum
    
    na_clttp = np.dot(na_mll_inv, na_cltt)
    na_clttp = na_clttp[2:] # remove monopole and dipole
    na_cltt = na_cltt[2:] # remove monopole and dipole

    # Load cl_curvature

    print ""
    print "Loading beam and cl_curvature..."

    na_bl = np.load(s_fn_beam)
    na_clcurv = np.loadtxt(s_fn_clcurv)

    # Load skewness and kurtosis power spectra

    if b_plot_skewness:

        print ""
        print "Loading skewness power spectrum..."

        na_cl21_data = np.loadtxt(s_fn_cl21_data)
        na_cl21_ana = np.loadtxt(s_fn_cl21_ana)

    if b_plot_kurtosis:

        print ""
        print "Loading kurtosis power spectra..."
        
        na_kl22_data = np.loadtxt(s_fn_kl22_data)
        na_kl22_ana = np.loadtxt(s_fn_kl22_ana)
        na_kl31_data = np.loadtxt(s_fn_kl31_data)
        na_kl31_ana = np.loadtxt(s_fn_kl31_ana)

    if b_plot_sims:

        print ""
        print "Loading Planck power spectrum and simulations..."

        i_nsims = 100
        i_sim = 0
        s_fn_cltt_sim = ('/data-1/jobryan/fnl_Planck/sims/na_cltt_sim_%i.npy' 
            % i_sim)

        na_cltt_planck = np.load(s_fn_cltt_planck)
        na_cltt_sim = np.load(s_fn_cltt_sim)
        na_cltt_sims = np.zeros((i_nsims, len(na_cltt_sim)))

        for i_sim in range(i_nsims):

            s_fn_cltt_sim = ('/data-1/jobryan/fnl_Planck/sims/na_cltt_sim_%i.npy' 
                % i_sim)
            na_cltt_sims[i_sim] = np.load(s_fn_cltt_sim)

        na_cltt_sims_avg = np.average(na_cltt_sims, axis=0)
        na_cltt_sims_std = np.std(na_cltt_sims, axis=0)

    if b_plot_alpha_beta:

        na_l, na_r, na_dr, na_alpha, na_beta = np.loadtxt(s_fn_alphabeta, 
                            usecols=(0,1,2,3,4), unpack=True, skiprows=3)

        na_l = np.unique(na_l)
        na_r = np.unique(na_r)[::-1]


        i_num_ell = len(na_l)
        i_num_r = len(na_r)

        na_alpha = na_alpha.reshape(i_num_ell, i_num_r)
        na_beta = na_beta.reshape(i_num_ell, i_num_r)
        na_dr = na_dr.reshape(i_num_ell, i_num_r)
        na_dr = na_dr[0]

        f_taustar = 235. #from Komatsu's thesis, section 4.2.2
        f_tau0 = 14297.6746786552 #from ../alphabeta_mod/transfer_function.txt
        na_rscale = np.array([0.6, 0.8, 1.0, 1.2, 1.4]) * f_taustar
        na_r_plot = np.ones(len(na_rscale)) * f_tau0 - na_rscale
        na_r_plot = closest_points(na_r, na_r_plot)
        f_norm = 10**(-10)

        s_plot_fn = 'plots/fig1_alpha_beta_vs_ell.png'

        plot_alpha_beta(na_alpha, na_beta, na_r, na_r_plot, na_l, na_cltt, 
                        f_norm, s_plot_fn)
        

    '''
    Associated plots: map; mask; masked map; power spectrum of mask, power 
        spectrum of map, masked map, and mode coupling corrected map
    '''

    i_num_ell = min([len(na_ell), len(na_cltt), len(na_bl), len(na_clcurv)])

    if b_plot_skewness:

        i_num_ell = min([i_num_ell, len(na_cl21_ana), len(na_cl21_data)])

    if b_plot_kurtosis:

        i_num_ell = min([i_num_ell, len(na_kl22_ana), len(na_kl22_data),
            len(na_kl31_ana), len(na_kl31_data)])

    na_ell = na_ell[:i_num_ell]

    if b_plot_main:

        print ""
        print "Plotting power spectra with i_num_ell = %i" % i_num_ell

        plot_ps([na_ell, na_ell], [na_cltt[:i_num_ell], na_clcurv[:i_num_ell]], 
            ['$C_\ell^{\\theta\\theta}$', '$C_\ell^{\zeta}$'], 
            s_ylabel='$C_\ell$', s_title='', 
            s_fn_plot='plots/fig_cltt_clcurv.png')

        plot_fn([na_ell], [na_bl[:i_num_ell]], [''], s_xlabel='$\ell$', 
            s_ylabel='$b_\ell$', s_title='', s_fn_plot='plots/nofig_beam.png')

        plot_ps([na_ell], [na_wll[:i_num_ell]], [''], s_ylabel='$W_\ell$', 
            s_title='', s_fn_plot='plots/fig_wl.png', b_units_on=False)

        plot_heatmap(na_mll, s_xlabel="$\ell$", s_ylabel="$\ell'$", 
            s_title="log$M_{\ell\ell'}$", s_fn_plot='plots/fig_mll.png', 
            b_log=True)

        plot_ps([na_ell, na_ell], [na_cltt[:i_num_ell], na_clttp[:i_num_ell]], 
            ['Masked', 'Masked, Corrected'], s_ylabel='$C_\ell$', s_title='', 
            s_fn_plot='plots/fig_masked_masked_corrected_ps.png')

    if b_plot_skewness:

        plot_ps([na_ell, na_ell], [na_cl21_ana[:i_num_ell], na_cl21_data[:i_num_ell]], 
            ['$C_{\ell,ana}^{(2,1)}$', '$C_{\ell,data}^{(2,1)}$'], 
            s_ylabel='$C_\ell^{(2,1)}$', s_title='', 
            s_fn_plot='plots/nofig_cl21.png')

    if b_plot_kurtosis:

        plot_ps([na_ell, na_ell], [na_kl22_ana[:i_num_ell], na_kl22_data[:i_num_ell]], 
            ['$K_{\ell,ana}^{(2,2)}$', '$K_{\ell,data}^{(2,2)}$'], 
            s_ylabel='$K_\ell^{(2,2)}$', s_title='', 
            s_fn_plot='plots/nofig_kl22.png')

        plot_ps([na_ell, na_ell], [na_kl31_ana[:i_num_ell], na_kl31_data[:i_num_ell]], 
            ['$K_{\ell,ana}^{(3,1)}$', '$K_{\ell,data}^{(3,1)}$'], 
            s_ylabel='$K_\ell^{(3,1)}$', s_title='', 
            s_fn_plot='plots/nofig_kl31.png')

    if b_plot_sims:

        i_lmax_tmp = min(len(na_cltt_planck), len(na_cltt_sims[0]))
        na_ell_tmp = np.arange(i_lmax_tmp)

        plot_ps([na_ell_tmp, na_ell_tmp], [na_cltt_planck[:i_lmax_tmp], 
            na_cltt_sims_avg[:i_lmax_tmp]], 
            ['$C_{\ell_Planck}$', '$C_{\ell,sim}$'], 
            s_ylabel='$C_\ell$', s_title='', 
            s_fn_plot='plots/fig_cltt_sims.png')

    if b_plot_misc:

        plot_ps([na_ell, na_ell], [na_cltt[:i_num_ell], na_clcurv[:i_num_ell]], 
            ['$C_\ell^{\\theta\\theta}$', '$C_\ell^{\zeta}$'], 
            s_ylabel='$C_\ell$', s_title='', 
            s_fn_plot='plots/fig_cltt_clcurv.png')

    return

if __name__ == '__main__':
    main()