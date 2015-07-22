'''
make_alms_with_fnl.py

Create alms using the Elsner maps (http://gavo.mpe.mpg.de/pub/Elsner/) for
alm_G and alm_NG to create:

alm = alm_G + fnl * alm_NG

where the input parameter to this program is fnl.

'''

import healpy as hp
import numpy as np

nsims = 1000
fnl = 0

for i in range(1,nsims+1):
    fn_alm = 'alm_fnl_%i_sim_%04d.fits' % (fnl, i)
    fn_cl = 'cl_fnl_%i_sim_%04d.fits' % (fnl, i)
    print 'making %s...' % fn_alm
    fn_alm_l = 'alm_l_%04d_v3.fits' % (i,)
    fn_alm_nl = 'alm_nl_%04d_v3.fits' % (i,)
    alm_g = hp.read_alm(fn_alm_l)
    alm_ng = hp.read_alm(fn_alm_nl)
    alm = alm_g + fnl * alm_ng
    hp.write_alm(fn_alm, alm)
    print 'making %s...' % fn_cl
    cl = hp.alm2cl(alm)
    np.savetxt(fn_cl, cl)