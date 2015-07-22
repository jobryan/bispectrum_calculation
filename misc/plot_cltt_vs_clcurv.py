'''
plot_cltt_vs_clcurv.py

Created on June 23, 2014
Updated on June 23, 2014

@author:    Jon O'Bryan

@contact:   jobryan@uci.edu

@summary:   Plots cltt and cl_curvature together.

@inputs:    na_clcurv_ell_1499.txt: Created from 
				fnl_Planck/alphabeta/compute_clcurv
            na_cltt: Created from fnl_Planck/code/cltt.py

@outputs:   Saved to fnl_Planck/code/plots/fig_cltt_clcurv.png

'''

# Python imports
import time
import pickle
import itertools as it

# 3rd party imports
import numpy as np
from matplotlib import pyplot as plt

def main():
	na_clcurv = np.loadtxt('../output/na_clcurv_ell_1499.txt')
	na_cltt = np.loadtxt('../output/na_cltt_ell_1499.txt')
	na_ell = np.arange(2,1501)

	plt.figure()
	plt.plot(na_clcurv*na_ell*(na_ell+1.)/(2.*np.pi),label=r'$C_{\ell}^{\zeta}$')
	plt.plot(na_cltt*na_ell*(na_ell+1.)/(2.*np.pi),label=r'$C_{\ell}^{\theta\theta}$')
	plt.xlabel(r'$\ell$', fontsize=20)
	plt.ylabel(r'$\ell(\ell+1)C_{\ell}/(2\pi)$', fontsize=20)
	plt.legend()
	plt.savefig('../plots/fig_cltt_clcurv.png')
	plt.show()

if __name__=='__main__':
	main()