import numpy as np
import healpy as hp
import helpers as h

nl = 1024
nside = 512

fnl = 30
nsim = 1

fn_map = 'data/fnl_sims/map_fnl_%i_sim_%i.fits' % (int(fnl), nsim)
map_fnl = hp.read_map(fn_map)
map_processed = hp.remove_dipole(map_fnl) / 1e6 / 2.7
alm_fnl = hp.map2alm(map_processed, lmax=nl)

cltt = hp.anafast(map_processed)

# load and resize things

l, r, dr, alpha, beta = np.loadtxt('data/l_r_alpha_beta.txt', usecols=(0,1,2,3,4), unpack=True, skiprows=3)

l = np.unique(l)
r = np.unique(r)[::-1]
l = l[:nl]

nr = len(r)
nl = len(l)

alpha = alpha.reshape(1499, nr)
beta = beta.reshape(1499, nr)
dr = dr.reshape(1499, nr)
dr = dr[0]

ir = 0

alm_data = alm_fnl

Alm = np.zeros(alm_data.shape[0],complex)
Blm = np.zeros(alm_data.shape[0],complex)
clAB2 = np.zeros(nl+1)
clABB = np.zeros(nl+1)

#fnl

Alm_data = hp.almxfl(alm_data, alpha[:nl,ir] / cltt[:nl])
Blm_data = hp.almxfl(alm_data, beta[:nl,ir] / cltt[:nl])

for li in xrange(2,nl):
    I = hp.Alm.getidx(nl,li,np.arange(min(nl,li)+1))
    Alm[I]=alpha[li-2][ir]*(alm_data[I])/cltt[li]
    Blm[I]=beta[li-2][ir]*(alm_data[I])/cltt[li]

# make cls and save them away
cltt_Alm = hp.alm2cl(Alm)
cltt_Blm = hp.alm2cl(Blm)
np.savetxt('debug/cltt_Alm.dat', cltt_Alm)
np.savetxt('debug/cltt_Blm.dat', cltt_Blm)

#An_data = hp.alm2map(Alm_data, nside=nside, fwhm=0.00145444104333, verbose=False)
#Bn_data = hp.alm2map(Blm_data, nside=nside, fwhm=0.00145444104333, verbose=False)
An_data = hp.alm2map(Alm_data, nside=nside, fwhm=0.00145444104333)
Bn_data = hp.alm2map(Blm_data, nside=nside, fwhm=0.00145444104333)
An = hp.alm2map(Alm, nside=nside)
Bn = hp.alm2map(Blm, nside=nside)

# make cls and save them away
cltt_An = hp.anafast(An)
cltt_Bn = hp.anafast(Bn)
np.savetxt('debug/cltt_An.dat', cltt_An)
np.savetxt('debug/cltt_Bn.dat', cltt_Bn)

B2lm_data = hp.map2alm(Bn_data*Bn_data, lmax=nl)
ABlm_data = hp.map2alm(An_data*Bn_data, lmax=nl)

B2lm = hp.map2alm(Bn*Bn, lmax=nl)
ABlm = hp.map2alm(An*Bn, lmax=nl)

# make cls and save them away
cltt_B2lm = hp.alm2cl(B2lm)
cltt_ABlm = hp.alm2cl(ABlm)
np.savetxt('debug/cltt_B2lm.dat', cltt_B2lm)
np.savetxt('debug/cltt_ABlm.dat', cltt_ABlm)

clAB2_data = hp.alm2cl(Alm_data, B2lm_data, lmax=nl)
clABB_data = hp.alm2cl(ABlm_data, Blm_data, lmax=nl)

for li in xrange(2,nl+1):
    I = hp.Alm.getidx(nl,li,np.arange(min(nl,li)+1))
    clAB2[li] = (Alm[I[0]]*B2lm[I[0]].conj()
            +2.*sum(Alm[I[1:]]*B2lm[I[1:]].conj()))/(2.0*li+1.0)
    clABB[li] = (Blm[I[0]]*ABlm[I[0]].conj()
            +2.*sum(Blm[I[1:]]*ABlm[I[1:]].conj()))/(2.0*li+1.0)

# make cls and save them away
np.savetxt('debug/clAB2.dat', clAB2)
np.savetxt('debug/clABB.dat', clABB)

result = (clAB2[1:] + 2 * clABB[1:]) * r[ir]**2. * dr[ir]

cl21 = result

# make cls and save them away
np.savetxt('debug/cl21.dat', cl21)

print "done!"