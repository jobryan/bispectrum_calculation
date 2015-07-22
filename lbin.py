from pylab import *

def lbin(cl,bin=10):
    sum = 0
    binnedcl = zeros(len(cl))
    for i in range(len(cl)/bin):
        for j in range(bin):
            sum += cl[j+i*bin]
        for j in range(bin):
            binnedcl[j+i*bin] = sum/float(bin)
        sum = 0
    return binnedcl

if __name__ == '__main__':
    cl = linspace(0,100,101)
    cl2 = lbin(cl)
    plot(cl,label='orig')
    plot(cl2,label='binned')
    xlabel('$l$',fontsize=20)
    ylabel('$C_l$',fontsize=20)
    legend(loc='upper left')
    show()
