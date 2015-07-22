import numpy as np
import healpy as hp

nl = 2000

map_smica = hp.read_map('data/CompMap_CMB-smica_2048_R1.11.fits')
mask = hp.read_map('data/CompMap_Mask_2048_R1.00.fits')
map_processed = hp.remove_dipole(map_smica) / 1e6 / 2.7 * mask
alm_data = hp.map2alm(map_processed)
cltt = hp.anafast(map_processed)

np.save('debug/cltt_not_corrected.npy', cltt)

mll = np.load('output/na_mll_2000_lmax.npy')
mll_inv = np.linalg.inv(mll)
cltt_corrected = np.dot(mll_inv, cltt[:nl])
alm_sim = hp.synalm(cltt_corrected[:nl]) #this is probably the issue...

# load and resize things

l, r, dr, alpha, beta = np.loadtxt('data/l_r_alpha_beta.txt', usecols=(0,1,2,3,4), unpack=True, skiprows=3)

lmax = 1499

mll = np.load('output/na_mll_1499_lmax.npy')
mll_inv = np.linalg.inv(mll)

l = np.unique(l)
r = np.unique(r)[::-1]
l = l[:lmax]

nr = len(r)
nl = len(l)

alpha = alpha.reshape(nl, nr)
beta = beta.reshape(nl, nr)
dr = dr.reshape(nl, nr)
dr = dr[0]

bl = np.load('output/na_bl.npy')

ir = 10

alm_data = alm_data[:hp.Alm.getsize(nl)]
alm_sim = alm_sim[:hp.Alm.getsize(nl)]
cltt_corrected = cltt_corrected[:nl]
bl = bl[:nl]

#data

Alm_data = hp.almxfl(alm_data, alpha[:,ir] / cltt_corrected * bl)
Blm_data = hp.almxfl(alm_data, beta[:,ir] / cltt_corrected * bl)

# make cls and save them away
cltt_data_Alm = hp.alm2cl(Alm_data)
cltt_data_Blm = hp.alm2cl(Blm_data)
np.savetxt('debug/cltt_data_Alm.dat', cltt_data_Alm)
np.savetxt('debug/cltt_data_Blm.dat', cltt_data_Blm)

An_data = hp.alm2map(Alm_data, nside=2048, fwhm=0.00145444104333, verbose=False)
Bn_data = hp.alm2map(Blm_data, nside=2048, fwhm=0.00145444104333, verbose=False)

# make cls and save them away
cltt_data_An = hp.anafast(An_data)
cltt_data_Bn = hp.anafast(Bn_data)
np.savetxt('debug/cltt_data_An.dat', cltt_data_An)
np.savetxt('debug/cltt_data_Bn.dat', cltt_data_Bn)

B2lm_data = hp.map2alm(Bn_data*Bn_data, lmax=nl)
ABlm_data = hp.map2alm(An_data*Bn_data, lmax=nl)

# make cls and save them away
cltt_data_B2lm = hp.alm2cl(B2lm_data)
cltt_data_ABlm = hp.alm2cl(ABlm_data)
np.savetxt('debug/cltt_data_B2lm.dat', cltt_data_B2lm)
np.savetxt('debug/cltt_data_ABlm.dat', cltt_data_ABlm)

clAB2_data = hp.alm2cl(Alm_data, B2lm_data, lmax=nl)
clABB_data = hp.alm2cl(ABlm_data, Blm_data, lmax=nl)

# make cls and save them away
np.savetxt('debug/clAB2_data.dat', clAB2_data)
np.savetxt('debug/clABB_data.dat', clABB_data)

result_data = (clAB2_data[1:] + 2 * clABB_data[1:]) * r[ir]**2. * dr[ir]

cl21_data = np.dot(mll_inv, result_data)

# make cls and save them away
np.savetxt('debug/cl21_data.dat', cl21_data)

#sim

Alm_sim = hp.almxfl(alm_sim, alpha[:,ir] / cltt_corrected * bl)
Blm_sim = hp.almxfl(alm_sim, beta[:,ir] / cltt_corrected * bl)

# make cls and save them away
cltt_sim_Alm = hp.alm2cl(Alm_sim)
cltt_sim_Blm = hp.alm2cl(Blm_sim)
np.savetxt('debug/cltt_sim_Alm.dat', cltt_sim_Alm)
np.savetxt('debug/cltt_sim_Blm.dat', cltt_sim_Blm)

An_sim = hp.alm2map(Alm_sim, nside=2048, fwhm=0.00145444104333, verbose=False)
Bn_sim = hp.alm2map(Blm_sim, nside=2048, fwhm=0.00145444104333, verbose=False)

# make cls and save them away
cltt_sim_An = hp.anafast(An_sim)
cltt_sim_Bn = hp.anafast(Bn_sim)
np.savetxt('debug/cltt_sim_An.dat', cltt_sim_An)
np.savetxt('debug/cltt_sim_Bn.dat', cltt_sim_Bn)

B2lm_sim = hp.map2alm(Bn_sim*Bn_sim, lmax=nl)
ABlm_sim = hp.map2alm(An_sim*Bn_sim, lmax=nl)

# make cls and save them away
cltt_sim_B2lm = hp.alm2cl(B2lm_sim)
cltt_sim_ABlm = hp.alm2cl(ABlm_sim)
np.savetxt('debug/cltt_sim_B2lm.dat', cltt_sim_B2lm)
np.savetxt('debug/cltt_sim_ABlm.dat', cltt_sim_ABlm)

clAB2_sim = hp.alm2cl(Alm_sim, B2lm_sim, lmax=nl)
clABB_sim = hp.alm2cl(ABlm_sim, Blm_sim, lmax=nl)

# make cls and save them away
np.savetxt('debug/clAB2_sim.dat', clAB2_sim)
np.savetxt('debug/clABB_sim.dat', clABB_sim)

result_sim = (clAB2_sim[1:] + 2 * clABB_sim[1:]) * r[ir]**2. * dr[ir]

cl21_sim = np.dot(mll_inv, result_sim)

# make cls and save them away
np.savetxt('debug/cl21_sim.dat', cl21_sim)

print "done!"