import numpy as np
from matplotlib import pyplot as plt

print "loading alm_data, alm_sim..."

nl = 1024
nside = 512

fnl = 30
nsim = 1

l = np.arange(nl)
lmult = l*(l+1.)/2./np.pi

# Plot 2: cl_Alm_fnl, cl_Blm_fnl
print "loading cltt_data_Alm, Blm and cltt_fnl_Alm, cltt_fnl_Blm and plotting..."

cltt_data_Alm = np.loadtxt('debug/cltt_Alm.dat')
cltt_data_Blm = np.loadtxt('debug/cltt_Blm.dat')

cltt_data_Alm = cltt_data_Alm[:nl]
cltt_data_Blm = cltt_data_Blm[:nl]

plt.semilogy(l, lmult*cltt_data_Alm, label=r'$C_{\ell,fnl}^{A_{\ell m}}$', color='r')
plt.semilogy(l, lmult*cltt_data_Blm, label=r'$C_{\ell,fnl}^{B_{\ell m}}$', color='g')

plt.legend(loc=4)
plt.ylabel(r"$\ell(\ell+1)C_{\ell}/(2\pi)$", fontsize=20)
plt.xlabel(r"$\ell$", fontsize=20)
plt.xlim([0,nl])
fn = "debug/plots/plot2_fnl.png"
plt.savefig(fn)
plt.clf()

# Plot 3: cl_An_data, cl_Bn_data vs. cl_An_sim, cl_Bn_sim
print "loading cltt_data_An, Bn and cltt_sim_An, Bn and plotting..."

cltt_data_An = np.loadtxt('debug/cltt_An.dat')
cltt_data_Bn = np.loadtxt('debug/cltt_Bn.dat')

cltt_data_An = cltt_data_An[:nl]
cltt_data_Bn = cltt_data_Bn[:nl]

plt.semilogy(l, lmult*cltt_data_An, label=r'$C_{\ell,fnl}^{A(n)}$', color='r')
plt.semilogy(l, lmult*cltt_data_Bn, label=r'$C_{\ell,fnl}^{B(n)}$', color='g')

plt.legend(loc=4)
plt.ylabel(r"$\ell(\ell+1)C_{\ell}/(2\pi)$", fontsize=20)
plt.xlabel(r"$\ell$", fontsize=20)
plt.xlim([0,nl])
fn = "debug/plots/plot3_fnl.png"
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

plt.semilogy(l, lmult*cltt_data_B2lm, label=r'$C_{\ell,fnl}^{B2_{\ell m}}$', color='r')
plt.semilogy(l, lmult*cltt_data_ABlm, label=r'$C_{\ell,fnl}^{AB_{\ell m}}$', color='g')

plt.legend(loc=4)
plt.ylabel(r"$\ell(\ell+1)C_{\ell}/(2\pi)$", fontsize=20)
plt.xlabel(r"$\ell$", fontsize=20)
plt.xlim([0,nl])
fn = "debug/plots/plot4_fnl.png"
plt.savefig(fn)
plt.clf()

# Plot 5: clAB2_data, clABB_data vs. clAB2_sim, clABB_sim
print "loading clAB2_data, clABB and clAB2_sim, clABB and plotting..."

clAB2_data = np.loadtxt('debug/clAB2.dat')
clABB_data = np.loadtxt('debug/clABB.dat')

clAB2_data = clAB2_data[:nl]
clABB_data = clABB_data[:nl]

plt.semilogy(l, lmult*clAB2_data, label=r'$C_{\ell,fnl}^{AB2}$', color='r')
plt.semilogy(l, lmult*clABB_data, label=r'$C_{\ell,fnl}^{ABB}$', color='g')

plt.legend(loc=4)
plt.ylabel(r"$\ell(\ell+1)C_{\ell}/(2\pi)$", fontsize=20)
plt.xlabel(r"$\ell$", fontsize=20)
plt.xlim([0,nl])
fn = "debug/plots/plot5_fnl.png"
plt.savefig(fn)
plt.clf()

# Plot 6: cl21_data vs. cl21_sim
print "loading cl21_data and cl21_sim and plotting..."

cl21_data = np.loadtxt('debug/cl21.dat')

cl21_data = cl21_data[:nl]

plt.semilogy(l, lmult*cl21_data, label=r'$C_{\ell,fnl}^{(2,1)}$', color='r')

plt.legend(loc=4)
plt.ylabel(r"$\ell(\ell+1)C_{\ell}/(2\pi)$", fontsize=20)
plt.xlabel(r"$\ell$", fontsize=20)
plt.xlim([0,nl])
fn = "debug/plots/plot6_fnl.png"
plt.savefig(fn)
plt.clf()