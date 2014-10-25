# cython: profile=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cython cimport view
from libc.math cimport exp, sqrt, M_PI
from clibint cimport *


init_libint_base()


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


cdef np.ndarray[double] RenormPrefactor(int lambda_n):
    """ This is a part of RenormPrefactor described in eq. (19)
        of libint manual.
        The loop structure over (a_nx, a_ny) is used to generate angular
        momentum indices in canonical LIBINT order for all members of a shell
        of angular momentum lambda_n.
    """
    cdef int i, a_nx, a_ny, an_x, an_y, an_z
    prefactor = np.ones(shape=((lambda_n+1)*(lambda_n+2)/2), dtype=np.double)
    i = 0
    for a_nx in range(lambda_n+1):
        for a_ny in range(a_nx+1):
            an_x = lambda_n-a_nx
            an_y = a_nx-a_ny
            an_z = a_ny

            norm_constant = fact2(2 * lambda_n - 1)
            norm_constant /= fact2(2 * an_x - 1) * fact2(2 * an_y - 1) * fact2(2 * an_z - 1)
            prefactor[i] = sqrt(norm_constant)
            i += 1
    return prefactor


cdef void Fm(int m, double x, double *buffer): # (13)
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
    """ Contracted Gausian Basis Function eq. (1)
        of libint manual.
    """
    cdef:
        int alpha_len # num's of primitives
        int lambda_n  # orbital quantum number of CGBF, sum of nx, ny, nz
        double *A     # coordinates CGBF centered at
        double *alpha # orbital exponents
        double *norm  # normalization coefficients (4)
        double *coef  # contraction coefficients (4)

    def __cinit__(self, cgbf):
        if cgbf.powers[1] + cgbf.powers[2] > 0:
            raise ValueError("Only first primitive function in a shell permitted")
        self.A = <double *>np.PyArray_DATA(cgbf.origin)
        self.lambda_n = cgbf.powers[0]
        self.alpha_len = len(cgbf.pexps)
        self.alpha = <double *>np.PyArray_DATA(cgbf.pexps)
        self.norm = <double *>np.PyArray_DATA(cgbf.pnorms)
        self.coef = <double *>np.PyArray_DATA(cgbf.coefs)


cdef class Libint:
    """
        mode = 0 - compute ERI
        mode = 1 - compute Deriv
        mode = 2 - compute r12
    """
    cdef:
        int max_num_prim_comb, max_am, sum_am
        int mode
        CGBF a, b, c, d

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

        self.max_num_prim_comb = self.a.alpha_len * self.b.alpha_len * self.c.alpha_len * self.d.alpha_len
        self.max_am = max(self.a.lambda_n, self.b.lambda_n, self.c.lambda_n, self.d.lambda_n)
        self.sum_am = self.a.lambda_n + self.b.lambda_n + self.c.lambda_n + self.d.lambda_n


cdef class ERI(Libint):
    cdef:
        Libint_t libint_data
        double dist2_AB, dist2_CD
        object shell
        int memory_allocated, memory_required

    def __cinit__(self, cgbf_a, cgbf_b, cgbf_c, cgbf_d):
        self.mode=0
        memory_required = libint_storage_required(self.max_am, self.max_num_prim_comb)
        memory_allocated = init_libint(&self.libint_data, self.max_am, self.max_num_prim_comb)
        if memory_required<>memory_allocated:
            raise ValueError("memory allocation error")

        self.dist2_AB = vec_dist2(self.a.A, self.b.A)
        self.dist2_CD = vec_dist2(self.c.A, self.d.A)
        self.compute_ERI()


    cdef void compute_ERI(self):
        for m in range(3):
            self.libint_data.AB[m] = self.a.A[m] - self.b.A[m]
            self.libint_data.CD[m] = self.c.A[m] - self.d.A[m]
        self.compute_primquartets()
        self.build_shell()


    cdef void compute_primquartets(self):
        for i in range(self.a.alpha_len):
            for j in range(self.b.alpha_len):
                for k in range(self.c.alpha_len):
                    for l in range(self.d.alpha_len):
                        self.compute_primitive_data(i, j, k, l)


    cdef void compute_primitive_data(self, int i, int j, int k, int l):
        cdef:
            int m
            double zeta, eta, rho
            double P[3]
            double Q[3]
            double W[3]
            prim_data *pdata

        pdata = &self.libint_data.PrimQuartet[(((i*self.b.alpha_len)+j)*self.c.alpha_len+k)*self.d.alpha_len+l]

        zeta = self.a.alpha[i] + self.b.alpha[j] # (7)
        eta = self.c.alpha[k] + self.d.alpha[l]  # (8)
        rho = zeta * eta / (zeta + eta)          # (9)

        for m in range(3):
            P[m] = (self.a.A[m] * self.a.alpha[i] + self.b.A[m] * self.b.alpha[j]) / zeta # (10)
            Q[m] = (self.c.A[m] * self.c.alpha[k] + self.d.A[m] * self.d.alpha[l]) / eta  # (11)
            W[m] = (P[m] * zeta + Q[m] * eta) / (zeta + eta)                              # (12)
            pdata.U[0][m] = P[m] - self.a.A[m]
            if self.mode==1:
                pdata.U[1][m] = P[m] - self.b.A[m] # libderiv
            elif self.mode ==2:
                pdata.U[1][m] = Q[m] - self.a.A[m] # lib12
            pdata.U[2][m] = Q[m] - self.c.A[m]
            if self.mode==1:
                pdata.U[3][m] = Q[m] - self.d.A[m] # libderiv
            elif self.mode ==2:
                pdata.U[3][m] = P[m] - self.c.A[m] # lib12
            pdata.U[4][m] = W[m] - P[m]
            pdata.U[5][m] = W[m] - Q[m]

        Ca = self.a.norm[i] * self.a.coef[i]
        Cb = self.b.norm[j] * self.b.coef[j]
        Cc = self.c.norm[k] * self.c.coef[k]
        Cd = self.d.norm[l] * self.d.coef[l]
        S12 = sqrt(M_PI / zeta) * (M_PI / zeta) * exp(- self.a.alpha[i] * self.b.alpha[j] / zeta * self.dist2_AB) # (15)
        S34 = sqrt(M_PI / eta) * (M_PI / eta) * exp(- self.c.alpha[k] * self.d.alpha[l] / eta * self.dist2_CD)   # (16)
        norm_coef = 2 * sqrt(rho / M_PI) * S12 * S34  * Ca * Cb * Cc * Cd

        Fm(self.sum_am, rho * vec_dist2(P, Q), pdata.F) #(13)
        for m in range(self.sum_am + self.mode + 1):
            pdata.F[m] *= norm_coef # (17)

        if self.mode>0:
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
        if self.mode==2:
            pdata.ss_r12_ss = pdata.F[0]/rho + vec_dist2(P, Q) * (pdata.F[0] - pdata.F[1]) # lib12


    cdef void build_shell(self):
        cdef int n
        cdef int cgbf_a_nfunc, cgbf_b_nfunc, cgbf_c_nfunc, cgbf_d_nfunc
        cdef double *eri

        if self.max_am==0:
            self.shell = np.zeros(shape=(1,1,1,1), dtype=np.double)
            for n in range(self.max_num_prim_comb):
                self.shell[0,0,0,0] += self.libint_data.PrimQuartet[n].F[0]
        else:
            eri = build_eri[self.a.lambda_n][self.b.lambda_n][self.c.lambda_n][self.d.lambda_n](&self.libint_data, self.max_num_prim_comb)

            cgbf_a_nfunc = (self.a.lambda_n+1)*(self.a.lambda_n+2)/2
            cgbf_b_nfunc = (self.b.lambda_n+1)*(self.b.lambda_n+2)/2
            cgbf_c_nfunc = (self.c.lambda_n+1)*(self.c.lambda_n+2)/2
            cgbf_d_nfunc = (self.d.lambda_n+1)*(self.d.lambda_n+2)/2
            view = <np.double_t [:cgbf_a_nfunc,:cgbf_b_nfunc,:cgbf_c_nfunc,:cgbf_d_nfunc]> eri

            self.shell = np.asarray(view.copy())
            if self.a.lambda_n > 1:
                self.shell *= RenormPrefactor(self.a.lambda_n).reshape(-1, 1, 1, 1)
            if self.b.lambda_n > 1:
                self.shell *= RenormPrefactor(self.b.lambda_n).reshape(1, -1, 1, 1)
            if self.c.lambda_n > 1:
                self.shell *= RenormPrefactor(self.c.lambda_n).reshape(1, 1, -1, 1)
            if self.d.lambda_n > 1:
                self.shell *= RenormPrefactor(self.d.lambda_n).reshape(1, 1, 1, -1)

    def __dealloc__(self):
        free_libint(&self.libint_data)

def Libint_ERI(cgbf_a, cgbf_b, cgbf_c, cgbf_d):
    return ERI(cgbf_a, cgbf_b, cgbf_c, cgbf_d).shell


cdef class Deriv1(Libint):
    cdef:
        Libderiv_t libderiv_data
        double dist2_AB, dist2_CD
        object deriv1
        int memory_allocated, memory_required
        int max_cart_class_size

    def __cinit__(self, cgbf_a, cgbf_b, cgbf_c, cgbf_d):
        self.mode=1
        self.max_cart_class_size = self.a.lambda_n * self.b.lambda_n * self.c.lambda_n * self.d.lambda_n
        memory_required = libderiv1_storage_required(self.max_am, self.max_num_prim_comb, self.max_cart_class_size)
        memory_allocated = init_libderiv1(&self.libderiv_data, self.max_am, self.max_num_prim_comb, self.max_cart_class_size)
        if memory_required<>memory_allocated:
            raise ValueError("memory allocation error")

        self.dist2_AB = vec_dist2(self.a.A, self.b.A)
        self.dist2_CD = vec_dist2(self.c.A, self.d.A)

    cdef void compute_deriv1(self):
        for m in range(3):
            self.libderiv1_data.AB[m] = self.a.A[m] - self.b.A[m]
            self.libderiv1_data.CD[m] = self.c.A[m] - self.d.A[m]
        self.compute_primquartets()
        self.build_deriv1()

    cdef void build_deriv1(self):

        cgbf_a_nfunc = (self.a.lambda_n+1)*(self.a.lambda_n+2)/2
        cgbf_b_nfunc = (self.b.lambda_n+1)*(self.b.lambda_n+2)/2
        cgbf_c_nfunc = (self.c.lambda_n+1)*(self.c.lambda_n+2)/2
        cgbf_d_nfunc = (self.d.lambda_n+1)*(self.d.lambda_n+2)/2

        build_deriv1_eri[self.a.lambda_n][self.b.lambda_n][self.c.lambda_n][self.d.lambda_n](&self.libderiv_data, self.max_num_prim_comb)
        #view = <np.double_t [:cgbf_a_nfunc,:cgbf_b_nfunc,:cgbf_c_nfunc,:cgbf_d_nfunc]> self.libderiv_data.ABCD
        #self.deriv1 = np.asarray(view.copy())

    def __dealloc__(self):
        free_libderiv(&self.libderiv_data)

def Libint_Deriv1(cgbf_a, cgbf_b, cgbf_c, cgbf_d):
    return Deriv1(cgbf_a, cgbf_b, cgbf_c, cgbf_d).deriv1
