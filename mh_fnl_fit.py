'''
mh_fnl_fit.py

Created on August 23, 2014
Updated on August 23, 2014

@author:    Jon O'Bryan

@contact:   jobryan@uci.edu

@summary:   Using the Metropolis-Hastings algorithm (http://en.wikipedia.org/
            wiki/Metropolis%E2%80%93Hastings_algorithm), for Markov chain Monte
            Carlo fitting, fit the fnl parameters using cl21_ana and 
            cl21_data (with cl21_data_g subtracted previously). 
            (Previously calculated cl21_ana.cpp, cl21_data.py, and 
            cl21_data_g.py). Use the following for the values prescribed in the 
            MH algorithm:

            state, current or otherwise (theta, theta'): configuration of taunl,
                gnl.
            prior distribution (p(theta)): uniform distribution between 
                mu-4sigma and mu+4sigma (where mu, sigma are taken from previous 
                values of taunl, gnl; e.g., gnl ~ 0.4e5 +/- 7.8e5, 
                taunl ~ 1.35e4 +/- 1.95e4)
            likelihood function (p(z|theta)): exp(-chi^2) where 
                chi^2 = sum((kl_data - kl_ana(taunl,gnl))/ sigma^2)^2
                where sigma^2 is the variance (per ell mode) of the simulations 
                used to remove kl_data_g.
            transition matrix(q(theta|theta')): gaussian random variable in 
                neighborhood (say 0.01 * theta) of current state (i.e., 
                q(x|x') = x + 0.01*x*normal(1,0)).
            acceptance function( min(1, prior(theta') p(z|theta') 
                transition(theta'|theta) / prior(theta) p(z|theta') 
                transition(theta|theta')) ): value between 0 and 1 based off of 
                transition and priors.

            The algorithm:

            (1) Choose initial state for Markov chain (theta_0) -- can be based
                off of "good guess" (e.g., previous results).
            (2) Randomly pick a state (theta') according to the transition
                function.
            (3) Choose a uniform random variable between 0 and 1 (u). If u >
                acceptance, reject new state; if u < acceptance, accept new 
                state.
            (4) Repeat from step 2 until a large set of states (N) is generated.
                Remove some initial set of states (burn-in) and choose some set
                of spaced samples (thinning) for your final distribution.

            Using the distribution:

            Find mean, std in each variable for value +/- std; use confidence 
            regions code to plot confidence regions.

@inputs:    Requires na_cltt.npy, cl21_data_g.py, kl_data_g.py

@outputs:   fig_gnl_taunl_kl22.png, fig_gnl_taunl_kl31.png

@command:   Run with 

            python mh_kl_fit.py

'''

# Python imports
import time

# 3rd party imports
import numpy as np
from matplotlib import pyplot as plt

'''
Main
'''

def main():
    
    '''
    Run parameters
    '''

    i_burnin = 1000
    i_thinning = 20
    i_total = 21000
    f_step = 0.01

    '''
    Load model and data
    '''

    i_rsteps = 50
    i_Lsteps = 40
    i_lmax = 1400

    na_kl31_data = np.loadatxt('output/na_kl31_data_%i_rsteps_%i_lmax.dat' 
        % (i_rsteps, i_lmax))

    for i_sim in range(i_nsims):
        na_kl31_data_g_sims = na_kl31_data_g_sims.vstack([na_kl31_data_g_sims, 
            np.loadtxt('output/na_kl22_data_g_sim_%i_%i_rsteps_%i_lmax.dat' % 
            (i_sim, i_rsteps, i_lmax))])
    na_kl31_data_g = np.average(na_kl31_data_g_sims, axis=0)
    na_kl31_data_c = na_kl31_data - na_kl31_data_g # remove gaussian component for connected piece only
    na_kl31_sim_error = np.std(na_kl31_data_g_sims, axis=0)

    na_kl31_ana_gnl = np.loadtxt('output/kl_31_ana_%i_rsteps_%i_Lsteps_%i_lmax_1_gnl_0_tnl.dat' 
        % (i_rsteps, i_Lsteps, i_lmax))
    na_kl31_ana_taunl = np.loadtxt('output/kl_31_ana_%i_rsteps_%i_Lsteps_%i_lmax_0_gnl_1_tnl.dat' 
        % (i_rsteps, i_Lsteps, i_lmax))
    na_kl31_ana_11 = np.loadtxt('output/kl_31_ana_%i_rsteps_%i_Lsteps_%i_lmax_1_gnl_1_tnl.dat' 
        % (i_rsteps, i_Lsteps, i_lmax))
    na_kl31_ana_cross = na_kl31_ana_11 - na_kl31_ana_gnl - na_kl31_ana_taunl

    na_kl22_data = np.loadatxt('output/na_kl22_data_%i_rsteps_%i_lmax.dat' 
        % (i_rsteps, i_lmax))

    for i_sim in range(i_nsims):
        na_kl22_data_g_sims = na_kl22_data_g_sims.vstack([na_kl22_data_g_sims, 
            np.loadtxt('output/na_kl22_data_g_sim_%i_%i_rsteps_%i_lmax.dat' % 
            (i_sim, i_rsteps, i_lmax))])
    na_kl22_data_g = np.average(na_kl22_data_g_sims, axis=0)
    na_kl22_data_c = na_kl22_data - na_kl22_data_g # remove gaussian component for connected piece only
    na_kl22_sim_error = np.std(na_kl22_data_g_sims, axis=0)

    na_kl22_ana_gnl = np.loadtxt('output/kl_22_ana_%i_rsteps_%i_Lsteps_%i_lmax_1_gnl_0_tnl.dat' 
        % (i_rsteps, i_Lsteps, i_lmax))
    na_kl22_ana_taunl = np.loadtxt('output/kl_22_ana_%i_rsteps_%i_Lsteps_%i_lmax_0_gnl_1_tnl.dat' 
        % (i_rsteps, i_Lsteps, i_lmax))
    na_kl22_ana_11 = np.loadtxt('output/kl_22_ana_%i_rsteps_%i_Lsteps_%i_lmax_1_gnl_1_tnl.dat' 
        % (i_rsteps, i_Lsteps, i_lmax))
    na_kl22_ana_cross = na_kl22_ana_11 - na_kl22_ana_gnl - na_kl22_ana_taunl

    '''
    (1) Choose initial state
    '''

    f_gnl_31 = 0.4e5
    f_taunl_31 = 1.35e4
    f_gnl_22 = 0.4e5
    f_taunl_22 = 1.35e4
    na_theta_31 = np.array([f_gnl_31, f_taunl_31])
    na_theta_22 = np.array([f_gnl_22, f_taunl_22])

    '''
    Run MCMC loop
    '''

    na_states_31 = na_theta_31
    na_states_22 = na_theta_22

    for i in range(i_total):

        '''
        (2) Randomly pick state, theta_new, according to transition function
        '''

        na_theta_31_new = na_theta + np.random.rand(len(na_theta)) * f_step
        f_gnl_31_new = na_theta_31_new[0]
        f_taunl_31_new = na_theta_31_new[1]

        na_theta_22_new = na_theta + np.random.rand(len(na_theta)) * f_step
        f_gnl_22_new = na_theta_22_new[0]
        f_taunl_22_new = na_theta_22_new[1]

        '''
        (3) Accept new state based off of uniform distribution
        '''

        na_kl31_ana = (na_kl31_ana_gnl * f_gnl_31 + na_kl31_ana_taunl 
            * f_taunl_31 + na_kl31_ana_cross * f_gnl_31 * f_taunl_31)

        f_likelihood_31 = np.exp(-np.sum((na_kl_data_c - na_kl31_ana)
            /na_kl31_sim_error**2)**2)

        na_kl31_ana_new = (na_kl31_ana_gnl * f_gnl_31_new + na_kl31_ana_taunl 
            * f_taunl_31_new + na_kl31_ana_cross * f_gnl_31_new 
            * f_taunl_31_new)

        f_likelihood_31_new = np.exp(-np.sum((na_kl_data_c - na_kl31_ana_new)
            /na_kl31_sim_error**2)**2)

        f_accept_31 = np.random.rand()

        if f_accept_31 > min(1, f_likelihood_31_new / f_likelihood_31):
            na_theta_31 = na_theta_31_new

        na_states_31 = np.vstack([na_states_31, na_theta_31])

        na_kl22_ana = (na_kl22_ana_gnl * f_gnl_22 + na_kl22_ana_taunl 
            * f_taunl_22 + na_kl22_ana_cross * f_gnl_22 * f_taunl_22)

        f_likelihood_22 = np.exp(-np.sum((na_kl_data_c - na_kl22_ana)
            /na_kl22_sim_error**2)**2)

        na_kl22_ana_new = (na_kl22_ana_gnl * f_gnl_22_new + na_kl22_ana_taunl 
            * f_taunl_22_new + na_kl22_ana_cross * f_gnl_22_new * f_taunl_22_new)

        f_likelihood_22_new = np.exp(-np.sum((na_kl_data_c - na_kl22_ana_new)
            /na_kl22_sim_error**2)**2)

        f_accept_22 = np.random.rand()

        if f_accept_22 > min(1, f_likelihood_22_new / f_likelihood_22):
            na_theta_22 = na_theta_22_new

        na_states_22 = np.vstack([na_states_22, na_theta_22])


    na_states_31_mod = na_states_31[i_burnin::i_thinning]
    na_states_22_mod = na_states_22[i_burnin::i_thinning]

    '''
    Plot states
    '''

if __name__=='__main__':
    main()