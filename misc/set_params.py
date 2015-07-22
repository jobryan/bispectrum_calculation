'''
set_params.py

Created on July 2, 2014
Updated on July 2, 2014

@author:    Jon O'Bryan

@contact:   jobryan@uci.edu

@summary:   Set parameters for get_params function used throughout Python code

@inputs:    None

@outputs:   Parameters pickle file

            d_params

            saved to data/params.pkl

'''

# Python imports
import pickle

'''
Main: Default run
'''

def main():

    # Set parameters

    i_lmax = 2000
    s_fn_mask = 'data/CompMap_Mask_2048_R1.00.fits'
    s_map_name = 'SMICA'
    s_fn_mll = 'output/na_mll_ell_2000.npy'
    i_nside = 2048
    s_fn_map = 'data/CompMap_CMB-smica_2048_R1.11.fits'
    s_fn_beam = 'output/na_bl.npy'
    s_fn_alphabeta = 'data/l_r_alpha_beta.txt'
    s_fn_cltt = 'output/na_cltt.npy'

    d_params = {}

    d_params['i_lmax'] = i_lmax
    d_params['i_nside'] = i_nside
    d_params['s_fn_map'] = s_fn_map
    d_params['s_map_name'] = s_map_name
    d_params['s_fn_mask'] = s_fn_mask
    d_params['s_fn_mll'] = s_fn_mll
    d_params['s_fn_beam'] = s_fn_beam
    d_params['s_fn_alphabeta'] = s_fn_alphabeta
    d_params['s_fn_cltt'] = s_fn_cltt

    s_fn_params = '../data/params.pkl'
    pickle.dump( d_params, open( s_fn_params, "wb" ) )

    print 'Parameters set as:'

    for s_key in d_params.keys():
        print s_key, '=', d_params[s_key]

    print 'and saved to', s_fn_params

    return

if __name__ == '__main__':
    main()