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

def main(nsim=1, fnl=0.0):

    nl = 1024
    nside_fnl = 512

    # Load map, mll
    print ""
    print "Loading alm_g, alm_ng and creating map..."

    fn_almg = ('data/fnl_sims/alm_l_%04d_v3.fits' % (nsim,))
    #fn_almg = ('data/fnl_sims/alm_l_%i.fits' % (nsim,))
    almg = hp.read_alm(fn_almg)
    #almg = almg[:hp.Alm.getsize(nl)]
    fn_almng = ('data/fnl_sims/alm_nl_%04d_v3.fits' % (nsim,))
    #fn_almng = ('data/fnl_sims/alm_nl_%i.fits' % (nsim,))
    almng = hp.read_alm(fn_almng)
    #almng = almng[:hp.Alm.getsize(nl)]
    alm = almg * (2.7e6) + fnl * almng * (2.7e6) # convert to units of uK to be consistent with other maps

    map_sim_fnl = hp.alm2map(alm, nside=nside_fnl)
    
    #print "Normalizing map..."
    #map_sim_fnl *= (1e6 * 2.7) # convert to units of uK to be consistent with other maps

    fn_map = 'data/fnl_sims/map_fnl_%i_sim_%i.fits' % (int(fnl), nsim)
    print "Writing map: %s" % fn_map
    hp.write_map(fn_map, map_sim_fnl)

if __name__ == '__main__':
    if (len(sys.argv) > 1):
        main(int(sys.argv[1]), int(sys.argv[2]))
    else:
        main()
