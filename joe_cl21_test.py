from numpy import *
import healpy as hp
import sys

LMAX = 1024
#lm = hp.Alm(LMAX)
fnl = 30
dR = 1.1337565

############## DEBUG ##############
nl = 1024
nside = 512
###################################

# Read in cl, alpha(r) and beta(r)
#l, cl = loadtxt('cl_wmap5_bao_sn.dat',usecols=(0,1),unpack=True)
#R, a, b = loadtxt("total.dat", usecols = (0,3,4), unpack=True)

############## DEBUG ##############
fnl = 30
nsim = 1

fn_map = '../data/fnl_sims/map_fnl_%i_sim_%i.fits' % (int(fnl), nsim)
map_fnl = hp.read_map(fn_map)
map_fnl /= (1e6 * 2.7)
map_processed = hp.remove_dipole(map_fnl)
alm_fnl = hp.map2alm(map_processed, lmax=nl)

#cl = hp.anafast(map_processed)
cl = load('../output/cltt_theory.npy')

# load and resize things

l, R, dR, a, b = loadtxt('../data/l_r_alpha_beta.txt', usecols=(0,1,2,3,4), unpack=True, skiprows=3)

l = unique(l)
R = unique(R)[::-1]
l = l[:nl]

nR = len(R)
nl = len(l)

a = a.reshape(1499, nR)
b = b.reshape(1499, nR)
dR = dR.reshape(1499, nR)
dR = dR[0]

###################################
    
# Put alpha and beta in a format condusive for r dependance
#a = a.reshape(500,1999)
#b = b.reshape(500,1999)
#R = R.reshape(500,1999)

# Initialize arrays
N = 0

#Read In alms and cls
alm = hp.read_alm('Maps/alm_l_'+str(N)+'.fits')
flm = hp.read_alm('Maps/alm_nl_'+str(N)+'.fits')
Alm = zeros(alm.shape[0],complex)
Blm = zeros(alm.shape[0],complex)
CAB2l = zeros((a.shape[0],LMAX+1))
CABBl = zeros((a.shape[0],LMAX+1))
clab2 = zeros(CAB2l.shape[1])
clabb = zeros(CAB2l.shape[1])

############## DEBUG ##############
alm_data = alm + fnl * flm
Alm_jon = zeros(alm_data.shape[0],complex)
Blm_jon = zeros(alm_data.shape[0],complex)
clAB2_jon = zeros(nl+1)
clABB_jon = zeros(nl+1)
###################################

r = 0
print r
for l in xrange(2,LMAX):
    I = hp.Alm.getidx(LMAX,l,arange(min(LMAX,l)+1))
    Alm[I]=a[l-2][r]*(alm[I]+fnl*flm[I])/cl[l]
    Blm[I]=b[l-2][r]*(alm[I]+fnl*flm[I])/cl[l]

############## DEBUG ##############
for li in xrange(2,nl):
    I = hp.Alm.getidx(nl,li,arange(min(nl,li)+1))
    Alm_jon[I]=a[li-2][r]*(alm_data[I])/cl[li]
    Blm_jon[I]=b[li-2][r]*(alm_data[I])/cl[li]
###################################

Amap  = hp.alm2map(Alm,512,lmax=LMAX)
Bmap  = hp.alm2map(Blm,512,lmax=LMAX)
B2map = Bmap*Bmap
ABmap = Amap*Bmap
B2lm  = hp.map2alm(B2map,lmax=LMAX)
ABlm  = hp.map2alm(ABmap,lmax=LMAX)

############## DEBUG ##############
An = hp.alm2map(Alm_jon, nside=nside)
Bn = hp.alm2map(Blm_jon, nside=nside)
B2lm_jon = hp.map2alm(Bn*Bn, lmax=nl)
ABlm_jon = hp.map2alm(An*Bn, lmax=nl)
###################################

for l in xrange(2,LMAX+1):
    I = hp.Alm.getidx(LMAX,l,arange(min(LMAX,l)+1))
    CAB2l[r][l] = (Alm[I[0]]*B2lm[I[0]].conj()
                +2.*sum(Alm[I[1:]]*B2lm[I[1:]].conj()))/(2.0*l+1.0)
for l in xrange(2,LMAX+1):
    I = hp.Alm.getidx(LMAX,l,arange(min(LMAX,l)+1))
    CABBl[r][l] = (Blm[I[0]]*ABlm[I[0]].conj()
                +2.*sum(Blm[I[1:]]*ABlm[I[1:]].conj()))/(2.0*l+1.0)

############## DEBUG ##############
for li in xrange(2,nl+1):
    I = hp.Alm.getidx(nl,li,arange(min(nl,li)+1))
    clAB2_jon[li] = (Alm_jon[I[0]]*B2lm_jon[I[0]].conj()
            +2.*sum(Alm_jon[I[1:]]*B2lm_jon[I[1:]].conj()))/(2.0*li+1.0)
    clABB_jon[li] = (Blm_jon[I[0]]*ABlm_jon[I[0]].conj()
            +2.*sum(Blm_jon[I[1:]]*ABlm_jon[I[1:]].conj()))/(2.0*li+1.0)
###################################

#for i in xrange(CAB2l.shape[1]):
#    for j in xrange(CAB2l.shape[0]):
#        clab2[i] += dR*R[j][i]*R[j][i]*CAB2l[j][i]
#        clabb[i] += dR*R[j][i]*R[j][i]*CABBl[j][i]

cloc = (clab2[r]+2*clabb[r]) * (R[r])**2. * dR[r]

############## DEBUG ##############
clAB2_jon = clAB2_jon[1:]
clABB_jon = clABB_jon[1:]

result_jon = zeros(nl, dtype='d')
result_jon += (clAB2_jon + 2 * clABB_jon) * (R[r])**2. * dR[r]
###################################

cltt_Alm_jon = hp.alm2cl(Alm_jon)
cltt_Blm_jon = hp.alm2cl(Blm_jon)
cltt_An = hp.anafast(An)
cltt_Bn = hp.anafast(Bn)
cltt_B2lm_jon = hp.alm2cl(B2lm_jon)
cltt_ABlm_jon = hp.alm2cl(ABlm_jon)
savetxt('../debug2/cltt_joe_Alm.dat', cltt_Alm_jon)
savetxt('../debug2/cltt_joe_Blm.dat', cltt_Blm_jon)
savetxt('../debug2/cltt_joe_An.dat', cltt_An)
savetxt('../debug2/cltt_joe_Bn.dat', cltt_Bn)
savetxt('../debug2/cltt_joe_B2lm.dat', cltt_B2lm_jon)
savetxt('../debug2/cltt_joe_ABlm.dat', cltt_ABlm_jon)
savetxt('../debug2/clAB2_joe.dat', clAB2_jon)
savetxt('../debug2/clABB_joe.dat', clABB_jon)

print ("r: %i" % R[r])
print ("alm_masked: ", alm_data)
print ("cl: ", cl)
print ("fn_map: ", fn_map)

print 'done'