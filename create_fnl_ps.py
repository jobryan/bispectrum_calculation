'''
create_fnl_ps.py

Created on November 20, 2014
Updated on November 20, 2014

@author:    Jon O'Bryan

@contact:   jobryan@uci.edu

@summary:   Create a power spectrum with a preset fnl value to test fnl 
            pipeline.

@inputs:    Load cltt, alpha, beta, and beam from pre-computed files
            (located in "output/na_cltt.npy", "data/l_r_alpha_beta.txt", and 
            "output/na_bl.npy" respectively)

            alm_l: Downloaded from (http://gavo.mpe.mpg.de/pub/Elsner/)
            alm_nl: Same

@outputs:   Power spectrum with preset fnl value

            na_cl_fnl_x

            where x is the value of fnl used in creating the power spectra.

@command:   python create_fnl_ps.py

'''

import healpy as hp
import numpy as np
from matplotlib import pyplot as plt

nsims = 5
fnl = 0.0
cls_fnl = []
for sim in range(1,nsims+1):
    alm_l = hp.read_alm('sims/alm_l_%i_v3.fits' % sim)
    alm_nl = hp.read_alm('sims/alm_nl_%i_v3.fits' % sim)
    alm = alm_l + fnl*alm_nl
    cls_fnl.append(hp.alm2cl(alm))
cl_fnl = np.average(cls_fnl, axis=0)
fnOut = 'sims/cl_fnl_%i.dat' % int(fnl)
np.savetxt(fnOut, cl_fnl)
print 'saved to %s' % fnOut
