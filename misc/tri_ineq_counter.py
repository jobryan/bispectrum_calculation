i_sum_counter = 0

i_lmax = 10

for L in range(10):
    for l1 in range(i_lmax):
        for l2 in range(i_lmax):
            if (abs(l1 - l2) <= L and abs(l1 + l2) >= L):
                for l3 in range(i_lmax):
                    for l4 in range(i_lmax):
                        if (abs(l3 - l4) <= L and abs(l3 + l4) >= L):
                            i_sum_counter += 1

print "i_sum_counter: %i (for L = %i)" % (i_sum_counter, L)