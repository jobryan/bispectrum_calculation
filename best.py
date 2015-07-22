from pylab import *
from scipy.optimize import leastsq
from lbin import lbin

def residuals(p, y, ay):
    fnl = p
    err = y - fnl*ay
    return err

fnl = 0
#nl = 1024
nl = 600
lmax = 600

x = linspace(50,550,6)
ell = arange(20,lmax,40)
#cth = loadtxt("joe/wandelt_noNoise.dat")
#cth = loadtxt("output/cl_21_ana_437_rsteps_1499_ellsteps.dat")
cth = loadtxt("output/cl21_ana_chang.txt", usecols=(1,), unpack=True)

l = arange(1,cth.shape[0]+1)
cth = lbin(cth*(2*l+1),bin=40)
cth = cth[:lmax]
Cth = cth[ell]

N=11
FNL=zeros(N)

runType = 'data' #'sim', 'data', 'fnl'

print runType

for i in range(1,N):
    l = arange(nl)
    #cloc = loadtxt("VNoiseNoMask0/clocV_"+ str(i)+"_fnl%i_nonoisenobeam.dat" % fnl)
    if runType == 'fnl':
        cloc = loadtxt("output/cl21_fnl_%i_sim_%i.dat" % (fnl, i))
    elif runType == 'sim':
        cloc = loadtxt("output/cl21_sim_%i.dat" % i)
    elif runType == 'data':
        cloc = loadtxt("output/cl21_data.dat")
    cloc = cloc[:nl]
    Cloc = lbin(cloc*(2*l+1),bin=40)
    Cloc = Cloc[:lmax]
    Cd = Cloc[ell]
    p0 = 0
    plsq = leastsq(residuals, p0, args=(Cd,Cth)) 
    FNL[i] = plsq[0]

print mean(FNL)
print std(FNL)
# hist(FNL,20)
# xlabel(r'$fnl = %i$' % fnl, fontsize=20)
# show()

# cl21_fnl

dl = 50
nbins = lmax / dl
ltmp = arange(dl/2,lmax,dl)
cl21s_binned = zeros((2, nbins, N))
for j,fnl in enumerate([0,100]):
    for i in range(1,N):
        #l = arange(nl)
        cl21 = loadtxt("output/cl21_fnl_%i_sim_%i.dat" % (fnl, i))
        cl21s_binned[j,:,i] = (lbin(cl21, bin=dl)[:lmax])[ltmp]

cl21s_avg_fnl_0 = average(cl21s_binned[0,:,:], axis=1)
cl21s_avg_fnl_100 = average(cl21s_binned[1,:,:], axis=1)
cl21s_std_fnl_0 = std(cl21s_binned[0,:,:], axis=1)
cl21s_std_fnl_100 = std(cl21s_binned[1,:,:], axis=1)

# cl21_sim

# dl = 50
# nbins = lmax / dl
# ltmp = arange(dl/2,lmax,dl)
cl21_sim_binned = zeros((nbins, N))
for i in range(1,N):
    #l = arange(nl)
    cl21 = loadtxt("output/cl21_sim_%i.dat" % i)
    cl21_sim_binned[:,i] = (lbin(cl21, bin=dl)[:lmax])[ltmp]

cl21_avg_sim = average(cl21_sim_binned, axis=1)
cl21_std_sim = std(cl21_sim_binned, axis=1)

#cl21_theory = loadtxt("joe/wandelt_noNoise.dat")
cl21_theory = loadtxt("output/cl21_ana_chang.txt", usecols=(1,), unpack=True)
cl21_theory_binned = (lbin(cl21_theory, bin=dl)[:lmax])[ltmp]

ell_plot = arange(dl/2,lmax,dl)
ell_sq = ell_plot * (ell_plot+1) / 2 / pi
errorbar(ell_plot[:nbins], ell_sq[:nbins] * cl21s_avg_fnl_0,
            yerr=ell_sq[:nbins] * cl21s_std_fnl_0, label='fnl=0')
errorbar(ell_plot[:nbins], ell_sq[:nbins] * cl21s_avg_fnl_100,
            yerr=ell_sq[:nbins] * cl21s_std_fnl_100, label='fnl=100')
errorbar(ell_plot[:nbins], ell_sq[:nbins] * cl21_avg_sim,
            yerr=ell_sq[:nbins] * cl21_std_sim, label='sim')
plot(ell_plot[:nbins], ell_sq[:nbins] * cl21_theory_binned * -7.21581393011e-06,
            label='cl21_th x fnl(0)')
plot(ell_plot[:nbins], ell_sq[:nbins] * cl21_theory_binned * 0.000212219534221,
            label='cl21_th x fnl(100)')
plot(ell_plot[:nbins], ell_sq[:nbins] * cl21_theory_binned * -2.94731143085e-05,
            label='cl21_th x sim')
ylabel(r"$\ell(\ell+1)/(2\pi) C_{\ell}^{(2,1)}$", fontsize=20)
xlabel(r"$\ell$", fontsize=20)
legend(loc=4)
show()