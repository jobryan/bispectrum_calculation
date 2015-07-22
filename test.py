from pylab import *
from healpy.fitsfunc import read_alm
from healpy.sphtfunc import alm2cl


almg = read_alm('/home/jobryan/joseph_fnl_sims/alm_l_0001_v3.fits')
almng = read_alm('/home/jobryan/joseph_fnl_sims/alm_nl_0001_v3.fits')
almng = 100*almng
cl1 = alm2cl(almg)
cl2 = alm2cl(almg+almng)
cl3 = alm2cl(almng)
dcl = cl2-cl1
l = arange(len(cl1))

semilogy(l**2*cl1/2.0/pi,label='cl1')
semilogy(l**2*cl2/2.0/pi,label='cl2')
semilogy(l**2*cl3/2.0/pi,label='cl3')
semilogy(l**2*dcl/2.0/pi,label='dcl')
xlim(0,1000)
#ylim(1.0e-11,1.0e-8)
legend(loc='best').draw_frame(0)
show()




