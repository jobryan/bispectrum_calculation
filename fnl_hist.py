'''
plot_cl21.py
'''

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import leastsq
import healpy as hp
import itertools as it
import helpers as h

#nl = 1024
nl =  500
nsims = 2
nsims_fnl = 10
fnls = [0, 10, 30]

nbins = 10

l = np.arange(nl) + 2.
lmult = l * (l+1) / 2 / np.pi

cl21_data = np.loadtxt('output/cl21_data.dat')
cl21_data = cl21_data[:nl]

cl21_sims = np.zeros((nsims, nl))
for nsim in range(nsims):
    tmp = np.loadtxt('output/cl21_sim_%i.dat' % nsim)
    cl21_sims[nsim] = tmp[:nl]
cl21_sims_avg = np.average(cl21_sims, axis=0)

# cl21_fnl_sims = np.zeros((len(fnls), nsims_fnl, nl))
# cl21_fnl_sims_avg = np.zeros((len(fnls), nl))
# for i, fnl in enumerate(fnls):
#     for nsim in range(nsims_fnl):
#         tmp = np.loadtxt('output/cl21_fnl_%i_sim_%i.dat' % (fnl, nsim+1))
#         cl21_fnl_sims[i, nsim] = tmp[:nl]
#     cl21_fnl_sims_avg[i] = np.average(cl21_fnl_sims[i], axis=0)


cl21_fnl_sims_joe = np.zeros((len(fnls), nsims_fnl, nl))
cl21_fnl_sims_avg_joe = np.zeros((len(fnls), nl))
for i, fnl in enumerate(fnls):
    for nsim in range(nsims_fnl):
        tmp = np.loadtxt('debug/joe_cl21/clocN_%i_fnl%i.dat' % (nsim, fnl))
        cl21_fnl_sims_joe[i, nsim] = tmp[:nl]
    cl21_fnl_sims_avg_joe[i] = np.average(cl21_fnl_sims_joe[i], axis=0)

# for i, fnl in enumerate(fnls):
#     line_styles = ["-","--","-.",":"]
#     linecycler = it.cycle(line_styles)
#     ls = next(linecycler)
#     marker_styles = [".","*","s","o"]
#     linecycler = it.cycle(marker_styles)
#     mc = next(linecycler)
#     plt.plot(h.bin_array(l, nbins), h.bin_array(lmult*cl21_fnl_sims_avg[i], nbins), 
#         linestyle='None', marker=mc, label=r'$C_{\ell,fnl=%i}^{(2,1)}$' % fnl)

# fnl_joe plotting
# for i, fnl in enumerate(fnls):
#     line_styles = ["-","--","-.",":"]
#     linecycler = it.cycle(line_styles)
#     ls = next(linecycler)
#     marker_styles = [".","*","s","o"]
#     linecycler = it.cycle(marker_styles)
#     mc = next(linecycler)
#     plt.plot(h.bin_array(l, nbins), h.bin_array(lmult*cl21_fnl_sims_avg_joe[i], nbins), 
#         linestyle='None', marker=mc, label=r'$C_{\ell,fnl=%i,joe}^{(2,1)}$' % fnl)

# plt.legend(loc=2)
# plt.ylabel(r"$\ell(\ell+1)C_{\ell}/(2\pi)$", fontsize=20)
# plt.xlabel(r"$\ell$", fontsize=20)
# plt.xlim([0, nl])
# plt.show()

#################################
# Histogram
#################################

nbins = 40

ell = np.arange(20,nl,nbins)
#cth = np.load('output/cltt_theory.npy')
#cth = cl21_fnl_sims_avg_joe[0]
#cth = np.loadtxt('joe/wandelt_noNoise.dat')
cth = np.loadtxt("output/cl_21_ana_437_rsteps_1499_ellsteps.dat")
l = np.arange(1, cth.shape[0]+1)
cth = h.lbin(cth*(2*l+1), bin=nbins)
cth = cth[:nl]
Cth = cth[ell]

FNL = np.zeros(nsims_fnl)

fnl = fnls[0]

for nsim in range(nsims_fnl):
    l = np.arange(nl)
    #cloc = np.loadtxt("clocV_"+ str(i)+"_fnl%i.dat" % fnl)
    #Cloc = h.lbin(cloc * (2*l+1), bin=nbins)
    Cloc = h.lbin(cl21_fnl_sims_joe[0][nsim] * (2*l+1), bin=nbins)
    Cloc = Cloc[:nl]
    Cd = Cloc[ell]
    p0 = 100
    plsq = leastsq(h.residuals, p0, args=(Cd, Cth))
    FNL[nsim] = plsq[0]
print np.mean(FNL)
print np.std(FNL)
plt.hist(FNL,20,label='fnl=%i' % fnls[0])
plt.legend(loc=2)
plt.show()



# FNL = np.zeros((len(fnls), nsims_fnl))

# for i, fnl in enumerate(fnls):
#     for nsim in range(nsims_fnl):
#         l = np.arange(nl)
#         Cloc = h.lbin(cl21_fnl_sims_joe[i][nsim] * (2*l+1), bin=nbins)
#         Cloc = Cloc[:nl]
#         Cd = Cloc[ell]
#         p0 = 100
#         plsq = leastsq(h.residuals, p0, args=(Cd, Cth))
#         FNL[i][nsim] = plsq[0]
#     print np.mean(FNL[i])
#     print np.std(FNL[i])
#     plt.hist(FNL[i],20,label='fnl=%i' % fnl)
# plt.legend(loc=2)
# plt.show()

