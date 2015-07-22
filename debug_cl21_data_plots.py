import numpy as np
from matplotlib import pyplot as plt

print "loading alm_data, alm_sim..."

nl = 1499
l = np.arange(nl)
lmult = l*(l+1.)/2./np.pi

#alm_data = hp.read_alm('output/na_alm_data.fits')
#alm_sim = hp.read_alm('debug/alm_sim.fits')

# Plot 1: cltt_data vs. cltt_sim
# print "calculating cltt_data, cltt_sim and plotting..."

# cltt_data = np.loadtxt('debug/cltt_data.dat')
# cltt_sim = np.loadtxt('debug/cltt_sim.dat')

# cltt_data = cltt_data[:nl]
# cltt_sim = cltt_sim[:nl]

# plt.semilogy(l, lmult*cltt_data, label=r'$C_{\ell,data}$', color='r')
# plt.semilogy(l, lmult*cltt_sim, label=r'$C_{\ell,sim}$', color='g')

# plt.legend(loc=4)
# plt.ylabel(r"$\ell(\ell+1)C_{\ell}/(2\pi)$", fontsize=20)
# plt.xlabel(r"$\ell$", fontsize=20)
# plt.xlim([0,nl])
# fn = "debug/plots/plot1.png"
# plt.savefig(fn)
# plt.clf()

# Plot 2: cl_Alm_data, cl_Blm_data vs. cl_Alm_sim, cl_Blm_sim
print "loading cltt_data_Alm, Blm and cltt_sim_Alm, Blm and plotting..."

cltt_data_Alm = np.loadtxt('debug/cltt_data_Alm.dat')
cltt_data_Blm = np.loadtxt('debug/cltt_data_Blm.dat')
cltt_sim_Alm = np.loadtxt('debug/cltt_sim_Alm.dat')
cltt_sim_Blm = np.loadtxt('debug/cltt_sim_Blm.dat')

cltt_data_Alm = cltt_data_Alm[:nl]
cltt_data_Blm = cltt_data_Blm[:nl]
cltt_sim_Alm = cltt_sim_Alm[:nl]
cltt_sim_Blm = cltt_sim_Blm[:nl]

plt.semilogy(l, lmult*cltt_data_Alm, label=r'$C_{\ell,data}^{A_{\ell m}}$', color='r')
plt.semilogy(l, lmult*cltt_data_Blm, label=r'$C_{\ell,data}^{B_{\ell m}}$', color='g')
plt.semilogy(l, lmult*cltt_sim_Alm, label=r'$C_{\ell,sim}^{A_{\ell m}}$', color='b')
plt.semilogy(l, lmult*cltt_sim_Blm, label=r'$C_{\ell,sim}^{B_{\ell m}}$', color='m')

plt.legend(loc=4)
plt.ylabel(r"$\ell(\ell+1)C_{\ell}/(2\pi)$", fontsize=20)
plt.xlabel(r"$\ell$", fontsize=20)
plt.xlim([0,nl])
fn = "debug/plots/plot2.png"
plt.savefig(fn)
plt.clf()

# Plot 3: cl_An_data, cl_Bn_data vs. cl_An_sim, cl_Bn_sim
print "loading cltt_data_An, Bn and cltt_sim_An, Bn and plotting..."

cltt_data_An = np.loadtxt('debug/cltt_data_An.dat')
cltt_data_Bn = np.loadtxt('debug/cltt_data_Bn.dat')
cltt_sim_An = np.loadtxt('debug/cltt_sim_An.dat')
cltt_sim_Bn = np.loadtxt('debug/cltt_sim_Bn.dat')

cltt_data_An = cltt_data_An[:nl]
cltt_data_Bn = cltt_data_Bn[:nl]
cltt_sim_An = cltt_sim_An[:nl]
cltt_sim_Bn = cltt_sim_Bn[:nl]

plt.semilogy(l, lmult*cltt_data_An, label=r'$C_{\ell,data}^{A(n)}$', color='r')
plt.semilogy(l, lmult*cltt_data_Bn, label=r'$C_{\ell,data}^{B(n)}$', color='g')
plt.semilogy(l, lmult*cltt_sim_An, label=r'$C_{\ell,sim}^{A(n)}$', color='b')
plt.semilogy(l, lmult*cltt_sim_Bn, label=r'$C_{\ell,sim}^{B(n)}$', color='m')

plt.legend(loc=4)
plt.ylabel(r"$\ell(\ell+1)C_{\ell}/(2\pi)$", fontsize=20)
plt.xlabel(r"$\ell$", fontsize=20)
plt.xlim([0,nl])
fn = "debug/plots/plot3.png"
plt.savefig(fn)
plt.clf()

# Plot 4: cl_B2lm_data, cl_ABlm_data vs. cl_B2lm_sim, cl_ABlm_sim
print "loading cltt_data_B2lm, ABlm and cltt_sim_B2lm, ABlm and plotting..."

cltt_data_B2lm = np.loadtxt('debug/cltt_data_B2lm.dat')
cltt_data_ABlm = np.loadtxt('debug/cltt_data_ABlm.dat')
cltt_sim_B2lm = np.loadtxt('debug/cltt_sim_B2lm.dat')
cltt_sim_ABlm = np.loadtxt('debug/cltt_sim_ABlm.dat')

cltt_data_B2lm = cltt_data_B2lm[:nl]
cltt_data_ABlm = cltt_data_ABlm[:nl]
cltt_sim_B2lm = cltt_sim_B2lm[:nl]
cltt_sim_ABlm = cltt_sim_ABlm[:nl]

plt.semilogy(l, lmult*cltt_data_B2lm, label=r'$C_{\ell,data}^{B2_{\ell m}}$', color='r')
plt.semilogy(l, lmult*cltt_data_ABlm, label=r'$C_{\ell,data}^{AB_{\ell m}}$', color='g')
plt.semilogy(l, lmult*cltt_sim_B2lm, label=r'$C_{\ell,sim}^{B2_{\ell m}}$', color='b')
plt.semilogy(l, lmult*cltt_sim_ABlm, label=r'$C_{\ell,sim}^{AB_{\ell m}}$', color='m')

plt.legend(loc=4)
plt.ylabel(r"$\ell(\ell+1)C_{\ell}/(2\pi)$", fontsize=20)
plt.xlabel(r"$\ell$", fontsize=20)
plt.xlim([0,nl])
fn = "debug/plots/plot4.png"
plt.savefig(fn)
plt.clf()

# Plot 5: clAB2_data, clABB_data vs. clAB2_sim, clABB_sim
print "loading clAB2_data, clABB and clAB2_sim, clABB and plotting..."

clAB2_data = np.loadtxt('debug/clAB2_data.dat')
clABB_data = np.loadtxt('debug/clABB_data.dat')
clAB2_sim = np.loadtxt('debug/clAB2_sim.dat')
clABB_sim = np.loadtxt('debug/clABB_sim.dat')

clAB2_data = clAB2_data[:nl]
clABB_data = clABB_data[:nl]
clAB2_sim = clAB2_sim[:nl]
clABB_sim = clABB_sim[:nl]

plt.semilogy(l, lmult*clAB2_data, label=r'$C_{\ell,data}^{AB2}$', color='r')
plt.semilogy(l, lmult*clABB_data, label=r'$C_{\ell,data}^{ABB}$', color='g')
plt.semilogy(l, lmult*clAB2_sim, label=r'$C_{\ell,sim}^{AB2}$', color='b')
plt.semilogy(l, lmult*clABB_sim, label=r'$C_{\ell,sim}^{ABB}$', color='m')

plt.legend(loc=4)
plt.ylabel(r"$\ell(\ell+1)C_{\ell}/(2\pi)$", fontsize=20)
plt.xlabel(r"$\ell$", fontsize=20)
plt.xlim([0,nl])
fn = "debug/plots/plot5.png"
plt.savefig(fn)
plt.clf()

# Plot 6: cl21_data vs. cl21_sim
print "loading cl21_data and cl21_sim and plotting..."

cl21_data = np.loadtxt('debug/cl21_data.dat')
cl21_sim = np.loadtxt('debug/cl21_sim.dat')

cl21_data = cl21_data[:nl]
cl21_sim = cl21_sim[:nl]

plt.semilogy(l, lmult*cl21_data, label=r'$C_{\ell,data}^{(2,1)}$', color='r')
plt.semilogy(l, lmult*cl21_sim, label=r'$C_{\ell,sim}^{(2,1)}$', color='g')

plt.legend(loc=4)
plt.ylabel(r"$\ell(\ell+1)C_{\ell}/(2\pi)$", fontsize=20)
plt.xlabel(r"$\ell$", fontsize=20)
plt.xlim([0,nl])
fn = "debug/plots/plot6.png"
plt.savefig(fn)
plt.clf()