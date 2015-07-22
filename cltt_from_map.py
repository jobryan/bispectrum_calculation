# Python imports
import time
import pickle
import itertools as it
import sys

# 3rd party imports
import numpy as np
import healpy as hp
from mpi4py import MPI

#
import helpers as h

def main(run_type='data', nsim=0):

    nl = 1499
    if (run_type == 'data'):
        fn_map = h._fn_map
        fn_cltt_out = 'output/cltt_data.dat'
    elif (run_type == 'sim'):
        fn_map = 'output/map_sim_%i.fits' % nsim
        fn_cltt_out = 'output/cltt_sim_%i.dat' % nsim
    
    map_in = hp.read_map(fn_map)
    mask = hp.read_map(h._fn_mask)
    fn_mll = 'output/na_mll_%i_lmax.npy' % nl
    mll = np.load(fn_mll)
    mll_inv = np.linalg.inv(mll)
    
    nside = hp.get_nside(map_in)
    
    map_in /= (1e6 * 2.7)
    map_in = hp.remove_dipole(map_in)
    map_masked = map_in * mask
    
    cltt_masked = hp.anafast(map_masked)
    cltt_masked = cltt_masked[:nl]
    cltt_corrected = np.dot(mll_inv, cltt_masked)

    print 'Saving cltt to %s' % fn_cltt_out
    np.savetxt(fn_cltt_out, cltt_corrected)

if __name__ == '__main__':
    if (len(sys.argv) > 1):
        main(sys.argv[1], int(sys.argv[2]))
    else:
        main()