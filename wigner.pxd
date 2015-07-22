cdef class wigner:
    cdef double fact[10000]
    cpdef double w3j(self,int l1, int l2, int l3)
    cpdef double w3jsq(self,int l1, int l2, int l3)
    cpdef double F(self,int l1, int l2, int l)
