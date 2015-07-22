'''
cltt.py

Created on June 17, 2014
Updated on June 17, 2014

@author:    Jon O'Bryan

@contact:   jobryan@uci.edu

@summary:   Count the number of non-zero ell bins given: triangle inequality, 
            namely,

            abs(l1 - l2) <= l3
            &&
            abs(l1 + l2) >= l3
            
@inputs:    i_ell_max

@outputs:   i_tot_ell

'''

# Python imports

# 3rd party imports
import numpy as np
from matplotlib import pyplot as plt

def main():
    li_ell_max = [2,10,100,200]
    i_count = 0

    for i_ell_max in li_ell_max:

        for i_l1 in range(i_ell_max):
            
            for i_l2 in range(i_ell_max):
                
                for i_l3 in range(i_ell_max):

                    #print "(l1,l2,l3): (%i, %i, %i)" % (i_l1, i_l2, i_l3)

                    if ((abs(i_l1 - i_l2) <= i_l3) and (abs(i_l1 + i_l2) >= i_l3)):

                        i_count += 1
                        #print "non-zero"

        print "Total number of non-zero ell bins: %i (for i_ell_max %i)" % (i_count, i_ell_max)


if __name__ == '__main__':
    main()