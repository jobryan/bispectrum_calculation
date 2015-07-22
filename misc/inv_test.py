# inv_test.py
# test the ability for numpy to take inverses correctly

import numpy as np
from matplotlib import pyplot as plt

min_size = 2
max_size = 1500
#max_size = 100
num_steps = 100
#num_steps = 10
sizes = []
diag_avg = []
diag_std = []
off_diag_avg = []
off_diag_std = []

for size in np.linspace(min_size, max_size, num_steps):
    if int((size *100. / max_size)) % 2 == 0: 
        print '%.0f%% done...' % (size * 100./ max_size)
    size = int(size)
    mat = np.random.randn(size,size)
    mat_inv = np.linalg.inv(mat)
    unit = np.dot(mat_inv, mat)
    sizes.append(size)
    diag_avg.append(np.average(np.diag(unit)))
    diag_std.append(np.std(np.diag(unit)))
    off_diag_avg.append(np.average(unit - np.eye(size) * np.diag(unit)))
    off_diag_std.append(np.std(unit - np.eye(size) * np.diag(unit)))

plt.errorbar(sizes, diag_avg, yerr=diag_std)
plt.title('Diagonal average and std')
plt.show()
plt.errorbar(sizes, off_diag_avg, yerr=off_diag_std)
plt.title('Off-diagonal average and std')
plt.show()
plt.plot(sizes, np.array(off_diag_avg) / np.array(diag_avg))
plt.title('Off-diagonal to diagonal ratio')
plt.show()

print "done"