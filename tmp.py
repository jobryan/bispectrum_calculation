import numpy as np
from matplotlib import pyplot as plt
from lbin import lbin

nsims = 11
lmax = 990

dl = 20
nbins = lmax / dl
ltmp = np.arange(dl/2, lmax, dl)

ell = np.arange(998) + 2
ellsq = ell * (ell + 1) / 2 / np.pi

cth = np.loadtxt("output/cl21_ana_chang.txt", usecols=(1,),unpack=True)
cdata = np.loadtxt("output/cl21_data.dat")
csims = np.zeros((nsims,1499))

for i in np.arange(nsims):
    csims[i] = np.loadtxt("output/cl21_sim_%i.dat" % i)
    
csim_binned = (lbin(np.average(csims, axis=0), bin=dl)[:lmax])[ltmp]

cth_binned = (lbin(cth, bin=dl)[:lmax])[ltmp]
cdata_binned = (lbin(cdata, bin=dl)[:lmax])[ltmp]

#plt.plot(ell[:lmax], ellsq[:lmax] * cth[:lmax], label='theory')
plt.plot(ell[ltmp], ellsq[ltmp] * cdata_binned, label='data')
plt.plot(ell[ltmp], ellsq[ltmp] * csim_binned, label='sim')
plt.ylabel(r"$\ell(\ell+1)/(2\pi) C_{\ell}^{(2,1)}$", fontsize=20)
plt.xlabel(r"$\ell$", fontsize=20)
plt.legend(loc=4)
plt.show()


