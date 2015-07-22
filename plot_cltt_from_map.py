import numpy as np
from matplotlib import pyplot as plt
import helpers as h
from lbin import lbin

nsims = 11
nl = 1499

dl = 20
nbins = nl / dl
ltmp = np.arange(dl/2, nl, dl)

l = np.arange(nl)
lsq = l * (l+1) / 2 / np.pi

fn_cltt_data = 'output/cltt_data.dat'
cltt_data = np.loadtxt(fn_cltt_data)

cltt_sims = np.zeros((nsims, nl))
for sim in range(1,nsims+1):
    fn_cltt_sim = 'output/cltt_sim_%i.dat' % sim
    cltt_sims[sim-1] = np.loadtxt(fn_cltt_sim)

cltt_sim_binned = (lbin(np.average(cltt_sims, axis=0), bin=dl)[:nl])[ltmp]

# h.plot_ps([l] * nsims, [lsq*cl for cl in cltt_sims] + [lsq*cltt_data], 
#           ['sim %i' % i for i in range(1,nsims+1)] + ['data'], 
#           s_ylabel='$\ell(\ell+1)/2\pi C_\ell$', s_title='', 
#         s_fn_plot='plots/cltts.png')

h.plot_ps([l[ltmp], l], [cltt_sim_binned, cltt_data], 
          ['sim average', 'data'], s_ylabel='$C_\ell$', 
          s_title='', s_fn_plot='plots/cltts.png')