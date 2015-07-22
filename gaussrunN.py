from numpy import *
import healpy as hp
import sys

LMAX = 500
#lm = hp.Alm(LMAX)
fnl = 100
dR = 1.1337565

# Read in cl, alpha(r) and beta(r)
l, cl = loadtxt('cl_wmap5_bao_sn.dat',usecols=(0,1),unpack=True)
R, a, b = loadtxt("total.dat", usecols = (0,3,4), unpack=True)
    
# Put alpha and beta in a format condusive for r dependance
a = a.reshape(500,1999)
b = b.reshape(500,1999)
R = R.reshape(500,1999)

# Initialize arrays
   
#for N in arange(int(sys.argv[1]),500,int(sys.argv[2])):
for N in xrange(0,10):
    print '#########################################################'
    print ""
    print N
    print ""
    print '#########################################################'
    
    #Read In alms and cls
    alm = hp.read_alm('Maps/alm_l_'+str(N)+'.fits')
    flm = hp.read_alm('Maps/alm_nl_'+str(N)+'.fits')
    Alm = zeros(hp.Alm.getsize(LMAX),complex)
    #Alm = zeros(alm.shape[0],complex)
    Blm = zeros(hp.Alm.getsize(LMAX),complex)
    #Blm = zeros(alm.shape[0],complex)
    CAB2l = zeros((a.shape[0],LMAX+1))
    CABBl = zeros((a.shape[0],LMAX+1))
    clab2 = zeros(CAB2l.shape[1])
    clabb = zeros(CAB2l.shape[1])
   
    for r in xrange(a.shape[0]):
        print r
        for l in xrange(2,LMAX):
            I = hp.Alm.getidx(LMAX,l,arange(min(LMAX,l)+1))
            Alm[I]=a[r][l-2]*(alm[I]+fnl*flm[I])/cl[l]
            Blm[I]=b[r][l-2]*(alm[I]+fnl*flm[I])/cl[l]

        ############## DEBUG ##############
        #Alm2 = hp.almxfl(alm + fnl*flm, a[r,:LMAX] / cl[1:])
        #Blm2 = hp.almxfl(alm + fnl*flm, b[r,:LMAX] / cl[1:])
        ###################################

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
    savetxt("NoNoiseNoMask/clocN_"+str(N)+"_fnl"+str(fnl)+".dat", cloc)

