import numpy as np

mll = np.load('../output/na_mll_1499_lmax.npy')
mll_inv = np.linalg.inv(mll)
for sim in range(100,354):
    inFn = '../output/na_cl21_data_g_sim_%i_no_mll.dat' % sim
    cl21_sim = np.loadtxt(inFn)
    cl21_sim_fixed = np.dot(mll_inv, cl21_sim)
    outFn = '../output/na_cl21_data_g_sim_%i.dat' % sim
    np.savetxt(outFn, cl21_sim_fixed)
    print 'saved to %s' % outFn