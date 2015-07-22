'''
plot_cl21.py
'''

import numpy as np
from matplotlib import pyplot as plt
import healpy as hp
import itertools as it
import helpers as h

nl = 1024
nsims = 1
nsims_fnl = 1
fnls = [10, 30]

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

cl21_fnl_sims = np.zeros((len(fnls), nsims_fnl, nl))
cl21_fnl_sims_avg = np.zeros((len(fnls), nl))
for i, fnl in enumerate(fnls):
    for nsim in range(nsims_fnl):
        tmp = np.loadtxt('output/cl21_fnl_%i_sim_%i.dat' % (fnl, nsim+1))
        cl21_fnl_sims[i, nsim] = tmp[:nl]
    cl21_fnl_sims_avg[i] = np.average(cl21_fnl_sims[i], axis=0)


cl21_fnl_sims_joe = np.zeros((len(fnls), nsims_fnl, nl))
cl21_fnl_sims_avg_joe = np.zeros((len(fnls), nl))
for i, fnl in enumerate(fnls):
    for nsim in range(nsims_fnl):
        tmp = np.loadtxt('debug/clocN_%i_fnl%i.dat' % (nsim, fnl))
        cl21_fnl_sims_joe[i, nsim] = tmp[:nl]
    cl21_fnl_sims_avg_joe[i] = np.average(cl21_fnl_sims_joe[i], axis=0)

# plt.semilogy(l, lmult*cl21_data, linestyle='-', color='g', 
#     label=r'$C_{\ell,data}^{(2,1)}$')
# plt.semilogy(l, lmult*cl21_sims_avg, linestyle='-.', color='b', 
#     label=r'$C_{\ell,sims}^{(2,1)}$')
for i, fnl in enumerate(fnls):
    line_styles = ["-","--","-.",":"]
    linecycler = it.cycle(line_styles)
    ls = next(linecycler)
    marker_styles = [".","*","s","o"]
    linecycler = it.cycle(marker_styles)
    mc = next(linecycler)
    #plt.semilogy(h.bin_array(l, nbins), h.bin_array(lmult*cl21_fnl_sims_avg[i], nbins), linestyle=ls,
    #    label=r'$C_{\ell,fnl=%i}^{(2,1)}$' % fnl)
    plt.plot(h.bin_array(l, nbins), h.bin_array(lmult*cl21_fnl_sims_avg[i], nbins), 
        linestyle='None', marker=mc, label=r'$C_{\ell,fnl=%i}^{(2,1)}$' % fnl)

for i, fnl in enumerate(fnls):
    line_styles = ["-","--","-.",":"]
    linecycler = it.cycle(line_styles)
    ls = next(linecycler)
    marker_styles = [".","*","s","o"]
    linecycler = it.cycle(marker_styles)
    mc = next(linecycler)
    #plt.semilogy(h.bin_array(l, nbins), h.bin_array(lmult*cl21_fnl_sims_avg_joe[i], nbins), linestyle=ls,
    #    label=r'$C_{\ell,fnl=%i,joe}^{(2,1)}$' % fnl)
    plt.plot(h.bin_array(l, nbins), h.bin_array(lmult*cl21_fnl_sims_avg_joe[i], nbins), 
        linestyle='None', marker=mc, label=r'$C_{\ell,fnl=%i,joe}^{(2,1)}$' % fnl)

plt.legend(loc=2)
plt.ylabel(r"$\ell(\ell+1)C_{\ell}/(2\pi)$", fontsize=20)
plt.xlabel(r"$\ell$", fontsize=20)
plt.xlim([0, nl])
fn_cl21_plots = "plots/cl21_comparison.png"
plt.savefig(fn_cl21_plots)
#plt.show()
plt.clf()
