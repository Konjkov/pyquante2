# cython: profile=False

import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cython cimport view
from libc.math cimport exp, pow, sqrt, M_PI
from clibint cimport *


cdef ang(int i, max_am, int[3] result):
    """ Return n-th set of angular momentum indeces
        for shell with angular momentum = max_am, in
        the canonical LIBINT order.
        Read libint manual for detail.
    """
    cdef int k, l
    k = int((sqrt(8*i+1)-1)/2)
    l = k*(k+1)/2
    result[0] = max_am-k
    result[1] = l + k - i
    result[2] = i - l


cdef double vec_dist2(double a[3], double b[3]):
    """ Vector distance
    """
    return (a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]) + (a[2]-b[2])*(a[2]-b[2])


cdef double fact2(int k):
    """ Compute double factorial: k!! = 1*3*5*....k
        for any shell with angular momentum less then
        LIBINT_MAX_AM = 8 (maximum angular momentum + 1)
    """
    cdef double *fact2 = [1, # P shell
                          1*3, # D shell
                          1*3*5, # F shell
                          1*3*5*7, # G shell
                          1*3*5*7*9, # H shell
                          1*3*5*7*9*11, # I shell
                          1*3*5*7*9*11*13], # K shell
    if k>13:
        raise ValueError("k!! not implemented for k>13")
    elif k>0:
        return fact2[(k-1)/2]
    else:
        return 1.0


cdef Fm(int m, double x, double *buffer): # (13)
    """Compute list of Boys Functions: F0(x), F1(x)...Fm(x).
       Reference: B.A. Mamedov, Journal of Mathematical Chemistry July 2004,
       Volume 36, Issue 3, pp 301-306.
       On the Evaluation of Boys Functions Using Downward Recursion Relation.
       DOI 10.1023/B:JOMC.0000044226.49921.f5
    """
    cdef int k
    cdef double sum

    if (x<35.0):
        current = 1.0/(2*m + 1)
        sum = 0.0
        k = 0
        while current>1.0e-9:
            k += 1
            sum += current
            current *= (2 * x)/(2*m + 2*k + 1)
        buffer[m] = exp(-x) * sum
        # Downward Recursion
        for k in range(m,0,-1):
            buffer[k-1] = (2*x*buffer[k] + exp(-x))/(2*k-1)
    else:
        buffer[0] = sqrt(M_PI/x)/2.0
        # Upward Recursion
        for k in range(m):
          buffer[k+1] = ((2*k+1)*buffer[k] - exp(-x))/(2*x)


cdef class CGBF:
    """ Contracted Gausian Basis Function
    """
    cdef:
        int alpha_len
        int lambda_n
        double A[3]
        int N[3]
        double *alpha
        double *norm_coef

    def __cinit__(self, cgbf):
        for i in range(3):
            self.A[i] = cgbf.origin[i]
            self.N[i] = cgbf.powers[i]
        self.lambda_n = self.N[0] + self.N[1] + self.N[2]
        self.alpha_len = len(cgbf.pexps)
        self.alpha = <double *> PyMem_Malloc(self.alpha_len * sizeof(double))
        self.norm_coef = <double *> PyMem_Malloc(self.alpha_len * sizeof(double))
        for i in range(self.alpha_len):
            self.alpha[i] = cgbf.pexps[i]
            self.norm_coef[i] = cgbf.pnorms[i] * cgbf.coefs[i]

    def __dealloc__(self):
        PyMem_Free(self.alpha)
        PyMem_Free(self.norm_coef)


cdef class Libint:
    cdef:
        Libint_t libint_data
        int memory_allocated, memory_required, max_num_prim_comb, max_am, sum_am
        CGBF a, b, c, d
        double dist2_AB, dist2_CD
        double *norm_S12
        double *norm_S34

    def __cinit__(self, cgbf_a, cgbf_b, cgbf_c, cgbf_d):
        self.a = CGBF(cgbf_a)
        self.b = CGBF(cgbf_b)
        self.c = CGBF(cgbf_c)
        self.d = CGBF(cgbf_d)
        if self.b.lambda_n > self.a.lambda_n:
            raise ValueError("<ab|cd> violate Libint angular momentum permission: lambda(a)>=lambda(b)")
        if self.d.lambda_n > self.c.lambda_n:
            raise ValueError("<ab|cd> violate Libint angular momentum permission: lambda(c)>=lambda(d)")
        if self.a.lambda_n + self.b.lambda_n > self.c.lambda_n + self.d.lambda_n:
            raise ValueError("<ab|cd> violate Libint angular momentum permission: lambda(c)+lambda(d)>=lambda(a)+lambda(b)")
        init_libint_base()
        self.max_num_prim_comb = self.a.alpha_len * self.b.alpha_len * self.c.alpha_len * self.d.alpha_len
        self.max_am = max(self.a.lambda_n, self.b.lambda_n, self.c.lambda_n, self.d.lambda_n)
        self.sum_am = self.a.lambda_n + self.b.lambda_n + self.c.lambda_n + self.d.lambda_n
        memory_required = libint_storage_required(self.max_am, self.max_num_prim_comb)
        memory_allocated = init_libint(&self.libint_data, self.max_am, self.max_num_prim_comb)
        for i in xrange(3):
            self.libint_data.AB[i] = self.a.A[i] - self.b.A[i]
            self.libint_data.CD[i] = self.c.A[i] - self.d.A[i]

        self.dist2_AB = vec_dist2(self.a.A, self.b.A)
        self.dist2_CD = vec_dist2(self.c.A, self.d.A)

        self.norm_S12 = <double *> PyMem_Malloc(self.a.alpha_len * self.b.alpha_len * sizeof(double))
        for i in range(self.a.alpha_len):
            for j in range(self.b.alpha_len):
                Ca = self.a.norm_coef[i] # (4)
                Cb = self.b.norm_coef[j] # (4)
                zeta = self.a.alpha[i] + self.b.alpha[j]
                self.norm_S12[i*self.b.alpha_len+j] = pow(M_PI / zeta, 1.5) * exp(- self.a.alpha[i] * self.b.alpha[j] / zeta * self.dist2_AB) # (15)
                self.norm_S12[i*self.b.alpha_len+j] *= Ca * Cb

        self.norm_S34 = <double *> PyMem_Malloc(self.c.alpha_len * self.d.alpha_len * sizeof(double))
        for k in range(self.c.alpha_len):
            for l in range(self.d.alpha_len):
                Cc = self.c.norm_coef[k] # (4)
                Cd = self.d.norm_coef[l] # (4)
                eta = self.c.alpha[k] + self.d.alpha[l]
                self.norm_S34[k*self.d.alpha_len+l] = pow(M_PI / eta, 1.5) * exp(- self.c.alpha[k] * self.d.alpha[l] / eta * self.dist2_CD) # (16)
                self.norm_S34[k*self.d.alpha_len+l] *= Cc * Cd

        primitive_number = 0
        for i in range(self.a.alpha_len):
            for j in range(self.b.alpha_len):
                for k in range(self.c.alpha_len):
                    for l in range(self.d.alpha_len):
                        self.compute_primitive_data(i, j, k, l, &self.libint_data.PrimQuartet[primitive_number])
                        primitive_number += 1

    cdef double RenormPrefactor(self, s, p, q, r):
        """Actualy described in eq. (19) of libint manual.
        """
        cdef int res[3]

        norm_constant = 1
        ang(s, self.a.N[0], res)
        norm_constant *= fact2(2 * res[0] - 1) * fact2(2 * res[1] - 1) * fact2(2 * res[2] - 1)
        ang(p, self.b.N[0], res)
        norm_constant *= fact2(2 * res[0] - 1) * fact2(2 * res[1] - 1) * fact2(2 * res[2] - 1)
        ang(q, self.c.N[0], res)
        norm_constant *= fact2(2 * res[0] - 1) * fact2(2 * res[1] - 1) * fact2(2 * res[2] - 1)
        ang(r, self.d.N[0], res)
        norm_constant *= fact2(2 * res[0] - 1) * fact2(2 * res[1] - 1) * fact2(2 * res[2] - 1)
        norm_constant /= fact2(2 * self.a.lambda_n - 1)
        norm_constant /= fact2(2 * self.b.lambda_n - 1)
        norm_constant /= fact2(2 * self.c.lambda_n - 1)
        norm_constant /= fact2(2 * self.d.lambda_n - 1)
        return sqrt(1/norm_constant)


    cdef prim_data compute_primitive_data(self, int i, int j, int k, int l, prim_data *pdata):
        cdef:
            int m
            double zeta, eta, rho
            double P[3]
            double Q[3]
            double W[3]

        zeta = self.a.alpha[i] + self.b.alpha[j] # (7)
        eta = self.c.alpha[k] + self.d.alpha[l]  # (8)
        rho = zeta * eta / (zeta + eta)          # (9)

        for m in range(3):
            P[m] = (self.a.A[m] * self.a.alpha[i] + self.b.A[m] * self.b.alpha[j]) / zeta # (10)
            Q[m] = (self.c.A[m] * self.c.alpha[k] + self.d.A[m] * self.d.alpha[l]) / eta  # (11)
            W[m] = (P[m] * zeta + Q[m] * eta) / (zeta + eta)                              # (12)
            pdata.U[0][m] = P[m] - self.a.A[m]
            #pdata.U[1][m] = P[m] - B[m] # libderiv
            #pdata.U[1][m] = Q[m] - A[m] # lib12
            pdata.U[2][m] = Q[m] - self.c.A[m]
            #pdata.U[3][m] = Q[m] - D[m] # libderiv
            #pdata.U[3][m] = P[m] - C[m] # lib12
            pdata.U[4][m] = W[m] - P[m]
            pdata.U[5][m] = W[m] - Q[m]

        pdata.twozeta_a = 2 * self.a.alpha[i]
        pdata.twozeta_b = 2 * self.b.alpha[j]
        pdata.twozeta_c = 2 * self.c.alpha[k]
        pdata.twozeta_d = 2 * self.d.alpha[l]
        pdata.oo2z = 1.0 / (2 * zeta)
        pdata.oo2n = 1.0 / (2 * eta)
        pdata.oo2zn = 1.0 / (2 * (zeta + eta))
        pdata.poz = rho / zeta
        pdata.pon = rho / eta
        pdata.oo2p = 1.0 / (2 * rho)
        #pdata.ss_r12_ss = 0.0 # lib12

        Fm(self.sum_am, rho * vec_dist2(P, Q), pdata.F) #(13)
        for m in range(self.sum_am + 1):
            pdata.F[m] *= 2 * sqrt(rho / M_PI) * self.norm_S12[i*self.b.alpha_len+j] * self.norm_S34[k*self.d.alpha_len+l] # (17)

    def build_ERI(self):
        cdef int n, ijkl
        cdef int cgbf_a_nfunc, cgbf_b_nfunc, cgbf_c_nfunc, cgbf_d_nfunc
        cdef int s, p, q, r

        if self.a.N[1] + self.a.N[2] + self.b.N[1] + self.b.N[2] + \
           self.c.N[1] + self.c.N[2] + self.d.N[1] + self.d.N[2] >0:
            raise ValueError("Only first primitive function in a shell permitted")

        if self.max_am==0:
            result = np.zeros(shape=(1,1,1,1), dtype=np.double)
            for n in range(self.max_num_prim_comb):
                result[0,0,0,0] += self.libint_data.PrimQuartet[n].F[0]
            return result
        else:
            cgbf_a_nfunc = (self.a.lambda_n+1)*(self.a.lambda_n+2)/2
            cgbf_b_nfunc = (self.b.lambda_n+1)*(self.b.lambda_n+2)/2
            cgbf_c_nfunc = (self.c.lambda_n+1)*(self.c.lambda_n+2)/2
            cgbf_d_nfunc = (self.d.lambda_n+1)*(self.d.lambda_n+2)/2
            result = np.zeros(shape=(cgbf_a_nfunc, cgbf_b_nfunc, cgbf_c_nfunc, cgbf_d_nfunc), dtype=np.double)
            eri = build_eri[self.a.lambda_n][self.b.lambda_n][self.c.lambda_n][self.d.lambda_n](&self.libint_data, self.max_num_prim_comb)
            for s in range(cgbf_a_nfunc):
                for p in range(cgbf_b_nfunc):
                    for q in range(cgbf_c_nfunc):
                        for r in range(cgbf_d_nfunc):
                            ijkl = ((s*cgbf_b_nfunc+p)*cgbf_c_nfunc+q)*cgbf_d_nfunc+r
                            result[s,p,q,r] = eri[ijkl]
                            if self.max_am>1:
                                result[s,p,q,r] *= self.RenormPrefactor(s, p, q, r)
            return result

    def __dealloc__(self):
        free_libint(&self.libint_data)
        PyMem_Free(self.norm_S12)
        PyMem_Free(self.norm_S34)


def Permutable_ERI(cgbf_a, cgbf_b, cgbf_c, cgbf_d):
    swap_ab = sum(cgbf_b.powers) > sum(cgbf_a.powers)
    swap_cd = sum(cgbf_d.powers) > sum(cgbf_c.powers)
    swap_abcd = sum(cgbf_a.powers) + sum(cgbf_b.powers) > sum(cgbf_c.powers) + sum(cgbf_d.powers)

    if swap_ab:
        cgbf_a, cgbf_b = cgbf_b, cgbf_a
    if swap_cd:
        cgbf_c, cgbf_d = cgbf_d, cgbf_c
    if swap_abcd:
        cgbf_a, cgbf_b, cgbf_c, cgbf_d = cgbf_c, cgbf_d, cgbf_a, cgbf_b

    shell = Libint(cgbf_a, cgbf_b, cgbf_c, cgbf_d).build_ERI()

    if swap_abcd:
        shell = np.swapaxes(shell,0,2)
        shell = np.swapaxes(shell,1,3)
    if swap_cd:
        shell = np.swapaxes(shell,2,3)
    if swap_ab:
        shell = np.swapaxes(shell,0,1)
    return shell
