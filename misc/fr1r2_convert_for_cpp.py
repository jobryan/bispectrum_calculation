'''
fr1r2_convert_for_cpp.py

Created on June 24, 2014
Updated on June 24, 2014

@author:    Jon O'Bryan

@contact:   jobryan@uci.edu

@summary:   Convert l_r_r2_fr1r2.txt to formats readable by cpp code
            
@inputs:    l_r_r2_fr1r2.txt (located in fnl_Planck/code/data/; created by 
            running fnl_Planck/alphabeta_mod/compute_alphabeta)

            *NOTE: If this file needs to be cleaned -- i.e., there are entries 
            such as 0.45286-101 -- open it with Sublime Text 2 and do a search
            and replace using the regex to find (\.\d*)-(\d*) and replace with
            $1E-$2

@outputs:   fr1r2, ell, r, and dr files in formats readable by read_matrix (or
            read_3dvector for fr1r2) function used in kl_ana.cpp

            f_ell_[i_num_ell]_r1_[i_num_r]_r2_[i_num_r].txt
            ell_[i_num_ell].txt
            r_[i_num_r].txt
            dr_r_[i_num_r].txt

            saved to output/na_cltt.npy

'''

# Python imports

# 3rd party imports
import numpy as np

def main():

    s_fn_alphabeta = '../data/l_r_alpha_beta.txt'

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

    s_fn_ell = '../data/ell_%i' % i_num_ell
    s_fn_r = '../data/r_%i.txt' % i_num_r
    s_fn_dr = '../data/dr_r_%i.txt' % i_num_r
    s_fn_alpha = '../data/alpha_ell_%i_r_%i.txt' % (i_num_ell, i_num_r)
    s_fn_beta = '../data/beta_ell_%i_r_%i.txt' % (i_num_ell, i_num_r)

    np.savetxt(s_fn_ell, na_l)
    np.savetxt(s_fn_r, na_r)
    np.savetxt(s_fn_dr, na_dr)
    np.savetxt(s_fn_alpha, na_alpha)
    np.savetxt(s_fn_beta, na_beta)

    print "Done!"

if __name__=='__main__':
    main()