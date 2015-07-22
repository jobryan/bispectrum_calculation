'''
helpers.py
'''

'''
Get parameters
'''

# Python imports
import time
import pickle
import itertools as it

# 3rd party imports
import numpy as np
from matplotlib import pyplot as plt
import healpy as hp

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

def bin_array(na_arr, i_nbins):
    i_width = len(na_arr) / i_nbins
    na_mean = np.zeros(i_nbins)
    na_std = np.zeros(i_nbins)
    for i in range(i_nbins):
        na_slice = na_arr[i*i_width:(i+1)*i_width]
        na_mean[i] = np.average(na_slice)
    return na_mean

def bin_array_sq_sum(na_arr, i_nbins):
    i_width = len(na_arr) / i_nbins
    na_result_sq_sum = np.sqrt((na_arr[:(na_arr.size // i_width) 
        * i_width].reshape(-1, i_width)**2).sum(axis=1))
    return na_result_sq_sum

def bin_2d_arr(na_arr, i_nbins):
    # bins in the 2nd dimension
    i_nsims = np.shape(na_arr)[0]
    i_width = np.shape(na_arr)[1] / i_nbins
    na_mean = np.zeros(i_nbins)
    na_std = np.zeros(i_nbins)
    for i in range(i_nbins):
        na_slice = na_arr[:,i*i_width:(i+1)*i_width]
        na_mean[i] = np.average(na_slice)
        # na_std[i] = np.std(na_slice)
        na_std[i] = (np.sqrt(np.sum(
            [(np.std(na_slice[:,ell]))**2. for ell in range(i_width)])) 
            / np.sqrt(i_width))
        # na_std[i] = np.average(
        #     [np.std(na_slice[:,ell]) for ell in range(i_width)])
    return na_mean, na_std

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

def residuals(p, y, ay):
    fnl = p
    err = y - fnl*ay
    return err

def lbin(cl,bin=10):
    sum = 0
    binnedcl = np.zeros(len(cl))
    for i in range(len(cl)/bin):
        for j in range(bin):
            sum += cl[j+i*bin]
        for j in range(bin):
            binnedcl[j+i*bin] = sum/float(bin)
        sum = 0
    return binnedcl

#default vars
_fn_mask = 'yo'
_nl = 2000
_fn_beam = 'output/na_bl.npy'
_map_name = 'SMICA'
_fn_mask = 'data/CompMap_Mask_2048_R1.00.fits'
_fn_mll = 'output/na_mll_ell_2000.npy'
_nside = 2048
_fn_map = 'data/CompMap_CMB-smica_2048_R1.11.fits'
_fn_cltt = 'output/na_cltt.npy'
_fn_alphabeta = 'data/l_r_alpha_beta.txt'
