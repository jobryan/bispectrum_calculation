# Python imports
import time
import pickle
import itertools as it
import sys

# 3rd party imports
import numpy as np
from matplotlib import pyplot as plt
import healpy as hp

# Internal imports
import helpers as h

def main(nsim=0):

    nl = h._nl

    # Load map, mask, mll
    print ""
    print "Loading map, mask, mll, and calculating mll_inv..."

    map_data = hp.read_map(h._fn_map)
    mask = hp.read_map(h._fn_mask)
    mll = np.load(h._fn_mll)
    mll_inv = np.linalg.inv(mll)

    # Read in Planck map: normalize, remove mono-/dipole, mask
    print "Normalizing, removing mono-/dipole, and masking map..."
    map_masked = map_data * mask
    # Create cltt (cltt_data_masked) and correct it (cltt_data_corrected)
    print "Calculating cltt_data_masked, cltt_data_corrected..."
    cltt_data_masked = hp.anafast(map_masked)
    cltt_data_masked = cltt_data_masked[:nl]
    cltt_data_corrected = np.dot(mll_inv, cltt_data_masked)
    # Create simulation of map (map_sim) from cltt_data_corrected
    print "Creating and saving map_sim_%i..." % nsim
    map_sim = hp.synfast(cltt_data_corrected, h._nside)
    hp.write_map('output/map_sim_%i.fits' % nsim, map_sim)

if __name__ == '__main__':
    if (len(sys.argv) > 1):
        main(int(sys.argv[1]))
    else:
        main()