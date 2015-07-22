'''
gaussian_estimator.py

Created on July 22, 2014
Updated on July 22, 2014

@author:    Jon O'Bryan

@contact:   jobryan@uci.edu

@summary:   Wrapper function for cl21_data_g.py and kl_data_g.py: 

            Runs cl21_data_g or kl_data_g (depends on s_est_run)
            for nsims creating na_cl21_data_g_[i_sim] or 
            na_kl_data_g_[i_sim] for nsims

@inputs:    Requires na_cltt.npy, cl21_data_g.py, kl_data_g.py

@outputs:   Outputs output/na_cl21_data_g_[i_sim].npy or 
            output/na_kl_data_g_[i_sim].npy

@command:   Run with 

            python gaussian_estimator.py [s_est_run] [i_nsims] [i_size] [i_sim_start]

            e.g.,

            python gaussian_estimator.py cl21 100 22 0

'''

# Python imports
import time
import os
import subprocess
import sys

# 3rd party imports
import numpy as np

'''
Main
'''

def main(s_est_run='cl21', i_nsims=100, i_size=22, i_sim_start=0):

    #s_est_run = 'kl' # alternatively, 'kl'
    
    '''
    Start cl21/kl estimator code
    '''

    f_t1 = time.time()

    i_sim_start = 100

    for i_sim in range(i_sim_start, i_nsims+i_sim_start):

        print ""
        print "Running %s estimator for sim %i..." % (s_est_run, i_sim)

        if s_est_run == 'cl21':
            subprocess.call(["mpirun", "-np", str(i_size), "python", 
                "cl21_data_g.py", str(i_sim)])
        elif s_est_run == 'kl':
            i_num_r = 50
            i_lmax = 400
            subprocess.call(["mpirun", "-np", str(i_size), "python", 
                "kl_data_g.py", str(i_sim), str(i_lmax), str(i_num_r)])

        print ("Finished gaussian estimator %i/%i (elapsed time: %s)" % (i_sim,
            i_nsims, str(time.time() - f_t1)) )

if __name__=='__main__':
    if (len(sys.argv) > 1):
        main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    else:
        main()