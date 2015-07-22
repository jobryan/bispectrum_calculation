cdef extern from "math.h":
    double exp(double)
    double log(double)
    double sqrt(double)
cdef class wigner:

    #cdef double fact[10000]

    def __cinit__(self):
        cdef int i
        self.fact[0] = 0.0
        self.fact[1] = 0.0
        for i in range(2,10000):
            self.fact[i] = log(i) + self.fact[i-1]

    cpdef double w3j(self,int l1,int l2,int l3):
        # Calculates the wigner-3j symbol. Please see eq. 15 of 
        # http://mathworld.wolfram.com/Wigner3j-Symbol.html for more
        # details. 
        cdef int L, z1, z2, z3, z4, z5, z6, z7, z8
        L = l1 + l2 + l3

        z1 = L - 2*l1
        z2 = L - 2*l2
        z3 = L - 2*l3
        z4 = L/2
        z5 = z4 - l1
        z6 = z4 - l2
        z7 = z4 - l3
        z8 = L+1
        #if z1 < 0 or z2 < 0 or z3 < 0 or z5 < 0 or z6 < 0 or z7 < 0 or L%2 > 0:
        #    return 0.0

        return (-1)**(L/2)*exp(0.5*(self.fact[z1] + self.fact[z2]  \
                   + self.fact[z3] - self.fact[z8] + 2.0*(self.fact[z4] \
                   - self.fact[z5] - self.fact[z6] - self.fact[z7])))

    cpdef double w3jsq(self,int l1,int l2,int l3):
        # Calculates the wigner-3j symbol. Please see eq. 15 of 
        # http://mathworld.wolfram.com/Wigner3j-Symbol.html for more
        # details. 
        cdef int L, z1, z2, z3, z4, z5, z6, z7, z8
        L = l1 + l2 + l3

        z1 = L - 2*l1
        z2 = L - 2*l2
        z3 = L - 2*l3
        z4 = L/2
        z5 = z4 - l1
        z6 = z4 - l2
        z7 = z4 - l3
        z8 = L+1
        #if z1 < 0 or z2 < 0 or z3 < 0 or z5 < 0 or z6 < 0 or z7 < 0 or L%2 > 0:
        #    return 0.0

        return exp(self.fact[z1] + self.fact[z2]  \
                   + self.fact[z3] - self.fact[z8] + 2.0*(self.fact[z4] \
                   - self.fact[z5] - self.fact[z6] - self.fact[z7]))

    cpdef double F(self,int l1, int l2, int l):
        return sqrt((2.0*l1+1)*(2.0*l2+1)*(2.0*l+1)*0.07957747154) \
                *self.w3j(l1,l2,l)

