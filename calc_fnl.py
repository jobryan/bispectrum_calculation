'''
calc_fnl.py

Created on August 21, 2014
Updated on August 21, 2014

@author:    Jon O'Bryan

@contact:   jobryan@uci.edu

@summary:   Calculate fnl from cl21_data, cl21_data_g, and cl21_ana:
            
            fnl = (cl21_data - cl21_data_g_avg) / cl21_ana

            Also, using Joseph's 5-year paper (arXiv:0907.4051v1) (Eqn. 49),
            we find the chisq as:

            chisq = (yT - M . p) C^-1 (y - M . p)

            where

            y = (cl21_data - cl21_data_g_avg)
            p = fnl
            M = cl21_ana
            C = covariance matrix = cov of sims

            and then plot exp(-chisq) vs. fnl. Calculate the 1-sigma, 2-sigma
            values (from Numerical Recipes); also, see Eqns. 50-51 in the above
            paper:

            p = (MT . C^-1 . M)^-1 . MT . C^-1 .y
            dp^2 = (M . C^-1 . M)^-1

@inputs:    Load cl21_data, cl21_data_g, and cl21_ana from pre-computed files
            (located in "output/na_cl21_data.dat", 
            "output/na_cl21_sim_[i_sim].npy", and 
            "output/cl_21_ana_[i_num_r]_rsteps_[i_num_ell]_ellsteps.dat" 
            respectively)

            cl21_data: Created by cl21_data.py
            cl21_data_g: Created by cl21_data_g.py
            cl21_ana: Created by cl21_ana.cpp

@outputs:   fnl_avg, fnl_std

            saved to 

            output/fnl_[i_num_r]_rsteps_[i_num_ell]_ellsteps_[i_nsims]_sims.dat

@command:   python calc_fnl.py

'''

# Python imports
import pickle

# 3rd party imports
import numpy as np
from matplotlib import pyplot as plt

'''
Helper functions
'''

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

'''
Main: Default run
'''

def main():

    '''
    Set parameters
    '''
    s_fn_params = 'data/params.pkl'
    (i_lmax, i_nside, s_fn_map, s_map_name, s_fn_mask, s_fn_mll, s_fn_beam, 
        s_fn_alphabeta, s_fn_cltt) = get_params(s_fn_params)

    i_nsims = 2
    #i_nsims = 93
    i_num_r = 437
    #i_num_ell = 1024
    i_num_ell = 1499

    '''
    Load data
    '''
    #s_fn_data = "output/cl21_data.dat"
    #s_fn_data = "output/cl21_ps_smica.dat"
    s_fn_data =  "output/cl21_fnl_100_sim_1.dat"
    na_cl21_data = np.loadtxt(s_fn_data)
    na_cl21_sims = np.zeros((i_nsims, len(na_cl21_data)))
    for i_sim in range(i_nsims):
        s_fn_data_g = 'output/cl21_sim_%i.dat' % i_sim
        tmp = np.loadtxt(s_fn_data_g)
        na_cl21_sims[i_sim] = tmp[:len(na_cl21_data)]

    s_fn_ana = "output/cl_21_ana_%i_rsteps_%i_ellsteps.dat" % (i_num_r, i_num_ell)
    na_cl21_ana = np.loadtxt(s_fn_ana)
    na_cl21_ana = na_cl21_ana[:len(na_cl21_data)]

    #na_cl21_data *= 2.725**3 #don't multiply for cl21_fnl
    #na_cl21_sims *= 2.725**3

    #na_cl21_ana *= (2.375)**6

    na_cl21_sims_avg = np.average(na_cl21_sims, axis=0)
    na_cl21_sims_std = np.std(na_cl21_sims, axis=0)

    checkUsingGaussianSignal = False
    if checkUsingGaussianSignal:
        na_cl21_data = na_cl21_sims_avg

    useFnlSims = False
    if useFnlSims:
        fnl = 10000
        i_fnl_sims = 4
        i_fnl_length = 1024
        na_cl21_fnl_sims = np.zeros((i_fnl_sims, i_fnl_length))
        for i in range(1, i_fnl_sims):
            s_fn_cl21_fnl_sim = "data/cl21_fnl_sims/cl21_fnl_%i_sim_%04d.dat" % (fnl, i)
            na_cl21_fnl_sims[i] = np.loadtxt(s_fn_cl21_fnl_sim)
        na_cl21_fnl_sims_avg = np.average(na_cl21_fnl_sims, axis=0)
        na_cl21_fnl_sims_avg *= (2.725)**6.
        # na_cl21_sims_avg = na_cl21_fnl_sims_avg
        # na_cl21_sims = na_cl21_fnl_sims
    useFnlSims2 = False
    if useFnlSims2:
        fnl_2 = 0
        i_fnl_sims2 = 9
        na_cl21_fnl_sims_2 = np.zeros((i_fnl_sims2, i_fnl_length))
        for i in range(1, i_fnl_sims2):
            s_fn_cl21_fnl_sim = "data/cl21_fnl_sims/cl21_fnl_%i_sim_%04d.dat" % (fnl_2, i)
            na_cl21_fnl_sims_2[i] = np.loadtxt(s_fn_cl21_fnl_sim)
        na_cl21_fnl_sims_avg_2 = np.average(na_cl21_fnl_sims_2, axis=0)
        na_cl21_fnl_sims_avg_2 *= (2.725)**6.
    useFnlSims3 = False
    if useFnlSims3:
        fnl_3 = 100
        i_fnl_sims3 = 3
        na_cl21_fnl_sims_3 = np.zeros((i_fnl_sims3, i_fnl_length))
        for i in range(1, i_fnl_sims3):
            s_fn_cl21_fnl_sim = "data/cl21_fnl_sims/cl21_fnl_%i_sim_%04d.dat" % (fnl_3, i)
            na_cl21_fnl_sims_3[i] = np.loadtxt(s_fn_cl21_fnl_sim)
        na_cl21_fnl_sims_avg_3 = np.average(na_cl21_fnl_sims_3, axis=0)
        na_cl21_fnl_sims_avg_3 *= (2.725)**6.

    '''
    Calculate fnl
    '''
    na_fnl_data = (na_cl21_data - na_cl21_sims_avg)/na_cl21_ana
    na_fnl_sims = (na_cl21_sims - na_cl21_sims_avg)/na_cl21_ana
    na_fnl_std = np.std(na_fnl_sims, axis=0)

    '''
    Write to file
    '''
    s_fn_out = ("output/fnl_%i_rsteps_%i_ellsteps_%i_sims.dat" % 
        (i_num_r, i_num_ell, i_nsims))
    np.savetxt(s_fn_out, na_fnl_data)

    '''
    Bin data
    '''
    i_width = 10
    #i_lmax_trunc = 1499
    i_lmax_trunc = 1024
    i_lmin_trunc = 0
    i_nbins = (i_lmax_trunc-i_lmin_trunc)/i_width

    b_off_diagonal = True

    na_fnl_avg_bin = bin_array(na_fnl_data[i_lmin_trunc:i_lmax_trunc+1], i_nbins)
    na_fnl_std_bin = bin_array(na_fnl_std[i_lmin_trunc:i_lmax_trunc+1], i_nbins)
    na_fnl_std_bin /= np.sqrt(i_width)

    na_cl21_ana_bin_avg = bin_array(na_cl21_ana[i_lmin_trunc:i_lmax_trunc+1], 
        i_nbins)
    na_cl21_data_bin_avg = bin_array(na_cl21_data[i_lmin_trunc:i_lmax_trunc+1],
        i_nbins)
    
    if b_off_diagonal:
        na_cl21_sims_bin = np.array(
            [bin_array(sim[i_lmin_trunc:i_lmax_trunc+1], i_nbins) 
            for sim in na_cl21_sims])
        na_cl21_sims_bin_avg = np.average(na_cl21_sims_bin, axis=0)
    else:
        na_cl21_sims_bin_avg, na_cl21_sims_bin_std = bin_2d_arr(
            na_cl21_sims[:,i_lmin_trunc:i_lmax_trunc+1], i_nbins)

    '''
    Calculate sigma (std, not sigma(fnl))
    '''

    f_sigma = 1./np.sqrt(np.sum(1./na_fnl_std_bin**2.))

    '''
    Calculate chi-squared and goodness of fit
    '''

    f_fnl = 1.0
    i_nfnl = 1000
    f_fnl_min = -30
    f_fnl_max = -10
    na_fnl = np.zeros(i_nfnl)
    na_chisq = np.zeros(i_nfnl)

    y = (na_cl21_data_bin_avg - na_cl21_sims_bin_avg)[:i_nbins]
    M = na_cl21_ana_bin_avg[:i_nbins]
    if b_off_diagonal:
        C = np.cov(na_cl21_sims_bin.transpose(), ddof=0)
    else:
        sigma = na_cl21_sims_bin_std[:i_nbins]
        C = np.eye(len(sigma)) * sigma**2.
    Cinv = np.linalg.inv(C)

    for i, f_fnl in enumerate(np.linspace(f_fnl_min, f_fnl_max, i_nfnl)):
        p = f_fnl
        na_chisq[i] = np.dot((y.transpose() - M*p), np.dot(Cinv, (y - M*p)))
        na_fnl[i] = f_fnl

    '''
    Minimize chisq: best fit fnl and error bars
    '''

    f_fnl = np.dot(np.dot(np.dot(1./np.dot(np.dot(M.transpose(), Cinv), M), 
        M.transpose()), Cinv), y)
    dp2 = 1./np.dot(np.dot(M, Cinv),M)
    dp = np.sqrt(abs(dp2))
    f_dfnl = dp

    print ("(l_min = %i, l_max = %i, dl = %i, off_diagonal = %s)" 
        % (i_lmin_trunc, i_lmax_trunc, i_width, str(b_off_diagonal)))
    print "fnl = %.6f +/- %.6f" % (f_fnl, f_dfnl)

    '''
    Alternative method for fnl: integrate likelihood function
    '''
    na_likelihood = np.exp(-na_chisq/i_nbins)
    na_likelihood *= 1./np.sum(na_likelihood)
    i_peak = np.argmin(na_chisq)
    f_int_likelihood = 0.
    di = 0
    while di < (len(na_likelihood)-1)/2:
        di += 1
        f_int_likelihood = np.sum(na_likelihood[i_peak-di:i_peak+di+1])
        if f_int_likelihood >= 0.6827:
            break

    '''
    Plots
    '''
    # na_ell = (np.arange(len(na_cl21_ana[i_lmin_trunc:i_lmax_trunc])) 
    #     + i_lmin_trunc)
    na_ell = np.arange(i_lmax_trunc)

    b_plot_chisq = False
    b_plot_fnl_std = False
    b_plot_fnl = False
    b_plot_sigma = False
    b_plot_cl21_all = True

    # chisq
    if b_plot_chisq:
        plt.plot(na_fnl, abs(na_chisq), 'bo')
        plt.xlabel(r'$f_{NL}$', fontsize=20)
        plt.ylabel(r'$\left|\chi^2\right|$', fontsize=20)
        s_fn_chisq = ('plots/chisq_fnl_dl_%i_lmin_%i_lmax_%i.png' 
            % (i_width, i_lmin_trunc, i_lmax_trunc))
        plt.savefig(s_fn_chisq)
        #plt.show()
        plt.clf()

    # fnl std
    if b_plot_fnl_std:
        plt.plot(np.arange(len(na_fnl_std_bin))*i_width,na_fnl_std_bin/np.sqrt(5)); 
        plt.xlim([0,600]); 
        plt.ylim([0,40]); 
        plt.ylabel(r'std($f_{NL}$)'); 
        plt.xlabel(r'$\ell$'); 
        plt.show()
        plt.clf()

    # fnl
    if b_plot_fnl:
        b_log = False
        if (i_lmax_trunc-i_lmin_trunc)%i_width !=0:
            plt.errorbar((na_ell)[::i_width][:-1], 
                na_fnl_avg_bin, yerr=na_fnl_std_bin, fmt='o')
        else:
            plt.errorbar((na_ell)[::i_width], 
                na_fnl_avg_bin, yerr=na_fnl_std_bin, fmt='o')
        plt.plot(np.zeros(len(na_ell)), linestyle='-')
        if b_log:
            plt.yscale('log', nonposy='clip')
        plt.ylabel(r"$f_{NL}$", fontsize=20)
        plt.xlabel(r"$\ell$", fontsize=20)
        plt.xlim([i_lmin_trunc,i_lmax_trunc])
        #plt.ylim([-400, 400])
        plt.ylim([-7000, 7000])
        s_fn_fnl_fig = ("plots/fnl_dl_%i_lmin_%i_lmax_%i_xtrunc.png" 
            % (i_width, i_lmin_trunc, i_lmax_trunc))
        plt.savefig(s_fn_fnl_fig)
        #plt.show()
        plt.clf()

    # sigma(fnl)
    if b_plot_sigma:
        b_log = False
        na_sigma_fnl = np.array([1./np.sqrt(np.sum((na_ell[:i] * 2. + 1.) * 
            (na_cl21_ana[:i]))) for i in range(1,len(na_ell))])
        plt.plot(na_sigma_fnl, linestyle='-')
        if b_log:
            plt.yscale('log', nonposy='clip')
        plt.ylabel(r"$\sigma(f_{NL})$ $[C_{\ell}^{(2,1)}]$", fontsize=20)
        plt.xlabel(r"$\ell_{max}$", fontsize=20)
        plt.ylim([0,50])
        plt.xlim([i_lmin_trunc,i_lmax_trunc])
        s_fn_sigma_fnl_fig = "plots/fig_sigma_fnl.png"
        plt.savefig(s_fn_sigma_fnl_fig)
        #plt.show()
        plt.clf()

    # cl21_data, cl21_data_g, cl21_ana
    if b_plot_cl21_all:
        b_log = True
        if b_log:
            plt.semilogy(na_ell[:i_fnl_length], na_ell[:i_fnl_length]*
                (na_ell[:i_fnl_length]+1.)/2.0/np.pi*na_cl21_sims_avg[:i_fnl_length], 
                label='sim, avg', color='r')
            plt.semilogy(na_ell[:i_fnl_length], na_ell[:i_fnl_length]*
                (na_ell[:i_fnl_length]+1.)/2.0/np.pi*na_cl21_data[:i_fnl_length], 
                label='data', color='g')
            plt.semilogy(na_ell[:i_fnl_length], na_ell[:i_fnl_length]*
                (na_ell[:i_fnl_length]+1.)/2.0/np.pi*na_cl21_ana[:i_fnl_length], 
                label='ana', color='b')
            # plt.plot(na_ell[:i_fnl_length], na_ell[:i_fnl_length]*
            #     (na_ell[:i_fnl_length]+1.)/2.0/np.pi*na_cl21_sims_avg[:i_fnl_length], 
            #     label='sim, avg', color='r')
            # plt.plot(na_ell[:i_fnl_length], na_ell[:i_fnl_length]*
            #     (na_ell[:i_fnl_length]+1.)/2.0/np.pi*na_cl21_data[:i_fnl_length], 
            #     label='data', color='g')
            # plt.plot(na_ell[:i_fnl_length], na_ell[:i_fnl_length]*
            #     (na_ell[:i_fnl_length]+1.)/2.0/np.pi*na_cl21_ana[:i_fnl_length], 
            #     label='ana', color='b')
            # if useFnlSims:
            #     plt.semilogy(na_ell[:i_fnl_length], na_ell[:i_fnl_length]
            #         *(na_ell[:i_fnl_length]+1.)/2.0/np.pi*na_cl21_fnl_sims_avg, 
            #         label='fnl = %i' % fnl, color='m')
            # if useFnlSims2:
            #     plt.semilogy(na_ell[:i_fnl_length], na_ell[:i_fnl_length]
            #         *(na_ell[:i_fnl_length]+1.)/2.0/np.pi*na_cl21_fnl_sims_avg_2, 
            #         label='fnl = %i' % fnl_2, color='k')
            # if useFnlSims3:
            #     plt.semilogy(na_ell[:i_fnl_length], na_ell[:i_fnl_length]
            #         *(na_ell[:i_fnl_length]+1.)/2.0/np.pi*na_cl21_fnl_sims_avg_3, 
            #         label='fnl = %i' % fnl_3, color='r')
        else:
            plt.plot(na_ell, na_ell*(na_ell+1.)/2.0/np.pi*na_cl21_sims_avg, 
                label='sim, avg', color='r')
            plt.plot(na_ell, na_ell*(na_ell+1.)/2.0/np.pi*na_cl21_data, label='data', 
                color='g')
            plt.plot(na_ell, na_ell*(na_ell+1.)/2.0/np.pi*na_cl21_ana, label='ana', 
                color='b')
        plt.legend(loc=4)
        plt.ylabel(r"$\ell(\ell+1)C_{\ell}/(2\pi)$", fontsize=20)
        plt.xlabel(r"$\ell$", fontsize=20)
        plt.xlim([i_lmin_trunc,i_lmax_trunc])
        #plt.yscale('symlog')
        s_fn_cl21_fig = "plots/cl21_comparison_tmp.png"
        plt.savefig(s_fn_cl21_fig)
        #plt.show()
        plt.clf()

if __name__ == '__main__':
    main()
