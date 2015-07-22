from numpy import *
#import healpy.fitsfunc as hpf
#import healpy.sphtfunc as hps
import healpy as hp
import sys

LMAX = 1024
#lm = hps.Alm(LMAX)
fnl = 0

l, cl = loadtxt('cl_wmap5_bao_sn.dat',usecols=(0,1),unpack=True)
#bl = loadtxt('bl_V.txt')
bl = ones(cl.shape[0])*1.0
     
# Read in alpha(r) and beta(r)
R, a, b = loadtxt("total.dat", usecols = (0,3,4), unpack=True)
    
# Put alpha and beta in a format condusive for r dependance
a = a.reshape(500,1999)
b = b.reshape(500,1999)
R = R.reshape(500,1999)

# Define A(r,lm) and B(r,lm)
#nl = ones(cl.shape[0])*2.39643128e-15
nl = ones(cl.shape[0])*0.0
nlm = hp.synalm(nl, lmax=LMAX)
 
dR = 1.1337565
   
#for N in xrange(111,121):    
for N in xrange(7,10):
    print '#########################################################'
    print ""
    print N
    print ""
    print '#########################################################'
    
    #Read In alms and cls
    alm = hp.read_alm('Maps/alm_l_'+str(N)+'.fits')
    flm = hp.read_alm('Maps/alm_nl_'+str(N)+'.fits')
    Alm = zeros(alm.shape[0],complex)
    Blm = zeros(alm.shape[0],complex)
    CAB2l = zeros((a.shape[0],LMAX+1))
    CABBl = zeros((a.shape[0],LMAX+1))
    clab2 = zeros(CAB2l.shape[1])
    clabb = zeros(CAB2l.shape[1])
    for r in xrange(a.shape[0]):
        print r
        for l in xrange(2,LMAX):
            I = hp.Alm.getidx(LMAX,l,arange(l+1))
            Alm[I]=a[r][l-2]*bl[l]*((alm[I]+fnl*flm[I])*bl[l]+nlm[I])/(cl[l]*bl[l]**2+nl[l])
            Blm[I]=b[r][l-2]*bl[l]*((alm[I]+fnl*flm[I])*bl[l]+nlm[I])/(cl[l]*bl[l]**2+nl[l])

        Amap  = hp.alm2map(Alm,512,lmax=LMAX)
        Bmap  = hp.alm2map(Blm,512,lmax=LMAX)
        B2map = Bmap*Bmap
        ABmap = Amap*Bmap
        B2lm  = hp.map2alm(B2map,lmax=LMAX)
        ABlm  = hp.map2alm(ABmap,lmax=LMAX)

        for l in xrange(2,LMAX+1):
            I = hp.Alm.getidx(LMAX,l,arange(min(LMAX,l)+1))
            CAB2l[r][l] = (Alm[I[0]]*B2lm[I[0]].conj()
                        +2.*sum(Alm[I[1:]]*B2lm[I[1:]].conj()))/(2.0*l+1.0)

        for l in xrange(2,LMAX+1):
            I = hp.Alm.getidx(LMAX,l,arange(min(LMAX,l)+1))
            CABBl[r][l] = (Blm[I[0]]*ABlm[I[0]].conj()
                        +2.*sum(Blm[I[1:]]*ABlm[I[1:]].conj()))/(2.0*l+1.0)
    
    for i in xrange(CAB2l.shape[1]):
        for j in xrange(CAB2l.shape[0]):
            clab2[i] += dR*R[j][i]*R[j][i]*CAB2l[j][i]
            clabb[i] += dR*R[j][i]*R[j][i]*CABBl[j][i]

    cloc = (clab2+2*clabb)
    savetxt("VNoiseNoMask0/clocV_"+str(N)+"_fnl"+str(fnl)+"_nonoisenobeam.dat", cloc)

