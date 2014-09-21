from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.math cimport exp, pow, sqrt, M_PI
from clibint cimport *


ctypedef struct cgbf:
    int alpha_len     # length of *alpha and *coef
    int lambda_n      # sum of N[i]
    double A[3]
    int N[3]
    double *alpha
    double *coef


cdef int ang_mom_index(int l, int m ,int n, int max_am):
    cdef int i
    i = 0
    for a_nx in range(max_am+1):
        for a_ny in range(a_nx+1):
            if (max_am-a_nx)==l and (a_nx-a_ny)==m and a_ny==n:
                return i
            i += 1


cdef double vec_dist2(double a[3], double b[3]):
    return (a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]) + (a[2]-b[2])*(a[2]-b[2])


cdef double Fm(int m, double x): # (13)
    cdef int k
    cdef double sum, current
    if (x<35.0):
        current = 1.0/(2*m + 1)
        sum = 0.0
        k = 0
        while current>1.0e-7:
            k += 1
            sum += current
            current *= (2 * x)/(2*m + 2*k + 1)
        return exp(-x) * sum
    else:
        current = sqrt(M_PI/x)/2.0
        for k in range(m):
          current = ((2*k-1)*current - exp(-x))/(2*x)
        return current


cdef int fact2(int k):
    cdef int i, result
    result = 1
    for i in range(1, k+1, 2):
        result *= i
    return result


cdef class CGBF:
    cdef:
        int alpha_len
        double A[3]
        int N[3]
        double *alpha
        double *coef

    def __cinit__(self, cgbf):
        for i in range(3):
            self.A[i] = cgbf.origin[i]
            self.N[i] = cgbf.powers[i]
        self.alpha_len = len(cgbf.pexps)
        self.alpha = <double *> PyMem_Malloc(self.alpha_len * sizeof(double))
        self.coef = <double *> PyMem_Malloc(self.alpha_len * sizeof(double))
        for i in range(self.alpha_len):
            self.alpha[i] = cgbf.pexps[i]
            self.coef[i] = cgbf.coefs[i]

    def __dealloc__(self):
        PyMem_Free(self.alpha)
        PyMem_Free(self.coef)


cdef class ERI:
    cdef:
        Libint_t libint_data
        int memory_allocated, memory_required, max_num_prim_comb, max_am
        cgbf cgbf_a
        cgbf cgbf_b
        cgbf cgbf_c
        cgbf cgbf_d

    def __cinit__(self, cgbf_a, cgbf_b, cgbf_c, cgbf_d):
        if cgbf_b.powers[0] + cgbf_b.powers[1] + cgbf_b.powers[2] > cgbf_a.powers[0] + cgbf_a.powers[1] + cgbf_a.powers[2]:
            cgbf_a, cgbf_b = cgbf_b, cgbf_a
        if cgbf_d.powers[0] + cgbf_d.powers[1] + cgbf_d.powers[2] > cgbf_c.powers[0] + cgbf_c.powers[1] + cgbf_c.powers[2]:
            cgbf_c, cgbf_d = cgbf_d, cgbf_c
        if cgbf_a.powers[0] + cgbf_a.powers[1] + cgbf_a.powers[2] + cgbf_b.powers[0] + cgbf_b.powers[1] + cgbf_b.powers[2] > \
           cgbf_c.powers[0] + cgbf_c.powers[1] + cgbf_c.powers[2] + cgbf_d.powers[0] + cgbf_d.powers[1] + cgbf_d.powers[2]:
            cgbf_a, cgbf_b, cgbf_c, cgbf_d = cgbf_c, cgbf_d, cgbf_a, cgbf_b
        for i in range(3):
            self.cgbf_a.A[i] = cgbf_a.origin[i]
            self.cgbf_b.A[i] = cgbf_b.origin[i]
            self.cgbf_c.A[i] = cgbf_c.origin[i]
            self.cgbf_d.A[i] = cgbf_d.origin[i]
            self.cgbf_a.N[i] = cgbf_a.powers[i]
            self.cgbf_b.N[i] = cgbf_b.powers[i]
            self.cgbf_c.N[i] = cgbf_c.powers[i]
            self.cgbf_d.N[i] = cgbf_d.powers[i]
        self.cgbf_a.alpha_len = len(cgbf_a.pexps)
        self.cgbf_b.alpha_len = len(cgbf_b.pexps)
        self.cgbf_c.alpha_len = len(cgbf_c.pexps)
        self.cgbf_d.alpha_len = len(cgbf_d.pexps)
        self.cgbf_a.alpha = <double *> PyMem_Malloc(self.cgbf_a.alpha_len * sizeof(double))
        self.cgbf_b.alpha = <double *> PyMem_Malloc(self.cgbf_b.alpha_len * sizeof(double))
        self.cgbf_c.alpha = <double *> PyMem_Malloc(self.cgbf_c.alpha_len * sizeof(double))
        self.cgbf_d.alpha = <double *> PyMem_Malloc(self.cgbf_d.alpha_len * sizeof(double))
        self.cgbf_a.coef = <double *> PyMem_Malloc(self.cgbf_a.alpha_len * sizeof(double))
        self.cgbf_b.coef = <double *> PyMem_Malloc(self.cgbf_b.alpha_len * sizeof(double))
        self.cgbf_c.coef = <double *> PyMem_Malloc(self.cgbf_c.alpha_len * sizeof(double))
        self.cgbf_d.coef = <double *> PyMem_Malloc(self.cgbf_d.alpha_len * sizeof(double))
        for i in range(self.cgbf_a.alpha_len):
            self.cgbf_a.alpha[i] = cgbf_a.pexps[i]
            self.cgbf_a.coef[i] = cgbf_a.coefs[i]
        for i in range(self.cgbf_b.alpha_len):
            self.cgbf_b.alpha[i] = cgbf_b.pexps[i]
            self.cgbf_b.coef[i] = cgbf_b.coefs[i]
        for i in range(self.cgbf_c.alpha_len):
            self.cgbf_c.alpha[i] = cgbf_c.pexps[i]
            self.cgbf_c.coef[i] = cgbf_c.coefs[i]
        for i in range(self.cgbf_d.alpha_len):
            self.cgbf_d.alpha[i] = cgbf_d.pexps[i]
            self.cgbf_d.coef[i] = cgbf_d.coefs[i]

        init_libint_base()
        self.max_num_prim_comb = self.cgbf_a.alpha_len * self.cgbf_b.alpha_len * self.cgbf_c.alpha_len * self.cgbf_d.alpha_len
        self.max_am = max((self.cgbf_a.N[0] + self.cgbf_a.N[1] + self.cgbf_a.N[2]),
                          (self.cgbf_b.N[0] + self.cgbf_b.N[1] + self.cgbf_b.N[2]),
                          (self.cgbf_c.N[0] + self.cgbf_c.N[1] + self.cgbf_c.N[2]),
                          (self.cgbf_d.N[0] + self.cgbf_d.N[1] + self.cgbf_d.N[2]))
        memory_required = libint_storage_required(self.max_am, self.max_num_prim_comb)
        memory_allocated = init_libint(&self.libint_data, self.max_am, self.max_num_prim_comb)
        for i in xrange(3):
            self.libint_data.AB[i] = self.cgbf_a.A[i] - self.cgbf_a.A[i]
            self.libint_data.CD[i] = self.cgbf_a.A[i] - self.cgbf_a.A[i]
        primitive_number = 0
        for i in range(self.cgbf_a.alpha_len):
            for j in range(self.cgbf_b.alpha_len):
                for k in range(self.cgbf_c.alpha_len):
                    for l in range(self.cgbf_d.alpha_len):
                        self.libint_data.PrimQuartet[primitive_number] = self.compute_primitive_data(i, j, k, l)
                        primitive_number += 1

    cdef double N(self, double alpha, int l, int m , int n): # normalization_constant (2)
        cdef double nominator, denominator
        nominator = pow(2, 2 * (l + m + n) + 1.5) * pow(alpha, l + m + n + 1.5)
        denominator = fact2(2 * l - 1) * fact2(2 * m - 1) * fact2(2 * n - 1) * pow(M_PI, 1.5)
        return sqrt(nominator/denominator)

    cdef prim_data compute_primitive_data(self, int i, int j, int k, int l):
        cdef:
            int m
            prim_data pdata
            double zeta, eta, rho
            double A[3]
            double B[3]
            double C[3]
            double D[3]
            double P[3]
            double Q[3]
            double W[3]
            double Ca, Cb, Cc, Cd
            double S12, S34

        zeta = self.cgbf_a.alpha[i] + self.cgbf_b.alpha[j] # (7)
        eta = self.cgbf_c.alpha[k] + self.cgbf_d.alpha[l]  # (8)
        rho = zeta * eta / (zeta + eta)                    # (9)

        for m in range(3):
            A[m] = self.cgbf_a.A[m]
            B[m] = self.cgbf_b.A[m]
            C[m] = self.cgbf_c.A[m]
            D[m] = self.cgbf_d.A[m]
            P[m] = (A[m] * self.cgbf_a.alpha[i] + B[m] * self.cgbf_b.alpha[j]) / zeta # (10)
            Q[m] = (C[m] * self.cgbf_c.alpha[k] + D[m] * self.cgbf_d.alpha[l]) / eta  # (11)
            W[m] = (P[m] * zeta + Q[m] * eta) / (zeta + eta)                          # (12)
            pdata.U[0][m] = P[m] - A[m]
            #pdata.U[1][m] = P[m] - B[m] # libderiv
            #pdata.U[1][m] = Q[m] - A[m] # lib12
            pdata.U[2][m] = Q[m] - C[m]
            pdata.U[3][m] = W[m] - P[m]
            #pdata.U[4][m] = Q[m] - D[m] # libderiv
            #pdata.U[4][m] = P[m] - C[m] # lib12
            pdata.U[5][m] = W[m] - Q[m]

        pdata.twozeta_a = 2 * self.cgbf_a.alpha[i]
        pdata.twozeta_b = 2 * self.cgbf_b.alpha[j]
        pdata.twozeta_c = 2 * self.cgbf_c.alpha[k]
        pdata.twozeta_d = 2 * self.cgbf_d.alpha[l]
        pdata.oo2z = 1.0 / (2 * zeta)
        pdata.oo2n = 1.0 / (2 * eta)
        pdata.oo2zn = 1.0 / (2 * (zeta + eta))
        pdata.poz = rho / zeta
        pdata.pon = rho / eta
        pdata.oo2p = 1.0 / (2 * rho)
        #pdata.ss_r12_ss = 0.0 # lib12

        Ca = self.cgbf_a.coef[i] * self.N(self.cgbf_a.alpha[i], self.cgbf_a.N[0]+self.cgbf_a.N[1]+self.cgbf_a.N[2], 0, 0) # (4)
        Cb = self.cgbf_b.coef[j] * self.N(self.cgbf_b.alpha[j], self.cgbf_b.N[0]+self.cgbf_b.N[1]+self.cgbf_b.N[2], 0, 0) # (4)
        Cc = self.cgbf_c.coef[k] * self.N(self.cgbf_c.alpha[k], self.cgbf_c.N[0]+self.cgbf_c.N[1]+self.cgbf_c.N[2], 0, 0) # (4)
        Cd = self.cgbf_d.coef[l] * self.N(self.cgbf_d.alpha[l], self.cgbf_d.N[0]+self.cgbf_d.N[1]+self.cgbf_d.N[2], 0, 0) # (4)
        S12 = pow(M_PI / zeta, 1.5) * exp(- self.cgbf_a.alpha[i] * self.cgbf_b.alpha[j] / zeta * vec_dist2(A, B)) # (15)
        S34 = pow(M_PI / eta, 1.5) * exp(- self.cgbf_c.alpha[k] * self.cgbf_d.alpha[l] / eta * vec_dist2(C, D))   # (16)
        m = self.cgbf_a.N[0] + self.cgbf_a.N[1] + self.cgbf_a.N[2] + \
            self.cgbf_b.N[0] + self.cgbf_b.N[1] + self.cgbf_b.N[2] + \
            self.cgbf_c.N[0] + self.cgbf_c.N[1] + self.cgbf_c.N[2] + \
            self.cgbf_d.N[0] + self.cgbf_d.N[1] + self.cgbf_d.N[2]
        for i in range(m+1):
            pdata.F[i] = 2 * Fm(i, rho * vec_dist2(P, Q)) * sqrt(rho / M_PI) * S12 * S34 * Ca * Cb * Cc * Cd # (17)
        return pdata


    def build_eri(self):
        cdef int n, ijkl
        cdef int a_lambda_n, b_lambda_n, c_lambda_n, d_lambda_n
        cdef int cgbf_a_nfunc, cgbf_b_nfunc, cgbf_c_nfunc, cgbf_d_nfunc
        cdef int s, p, q, r
        cdef double norm_constant, result
        if self.max_am==0:
            result = 0.0
            for n in range(self.max_num_prim_comb):
                result += self.libint_data.PrimQuartet[n].F[0]
        else:
            a_lambda_n = self.cgbf_a.N[0]+self.cgbf_a.N[1]+self.cgbf_a.N[2]
            b_lambda_n = self.cgbf_b.N[0]+self.cgbf_b.N[1]+self.cgbf_b.N[2]
            c_lambda_n = self.cgbf_c.N[0]+self.cgbf_c.N[1]+self.cgbf_c.N[2]
            d_lambda_n = self.cgbf_d.N[0]+self.cgbf_d.N[1]+self.cgbf_d.N[2]
            cgbf_a_nfunc = (a_lambda_n+1)*(a_lambda_n+2)/2
            cgbf_b_nfunc = (b_lambda_n+1)*(b_lambda_n+2)/2
            cgbf_c_nfunc = (c_lambda_n+1)*(c_lambda_n+2)/2
            cgbf_d_nfunc = (d_lambda_n+1)*(d_lambda_n+2)/2
            s = ang_mom_index(self.cgbf_a.N[0], self.cgbf_a.N[1], self.cgbf_a.N[2], a_lambda_n)
            p = ang_mom_index(self.cgbf_b.N[0], self.cgbf_b.N[1], self.cgbf_b.N[2], b_lambda_n)
            q = ang_mom_index(self.cgbf_c.N[0], self.cgbf_c.N[1], self.cgbf_c.N[2], c_lambda_n)
            r = ang_mom_index(self.cgbf_d.N[0], self.cgbf_d.N[1], self.cgbf_d.N[2], d_lambda_n)
            ijkl = ((s*cgbf_b_nfunc+p)*cgbf_c_nfunc+q)*cgbf_d_nfunc+r
            #print "s,p,q,r", s,p,q,r, "ijkl", ijkl
            result = build_eri[a_lambda_n][b_lambda_n][c_lambda_n][d_lambda_n](&self.libint_data, self.max_num_prim_comb)[ijkl]
            """for s in range(cgbf_a_nfunc):
                for p in range(cgbf_b_nfunc):
                    for q in range(cgbf_c_nfunc):
                        for r in range(cgbf_d_nfunc):
                            norm_constant = 1
                            norm_constant *= self.N(self.cgbf_a.alpha[0], self.cgbf_a.N[0], self.cgbf_a.N[1], self.cgbf_a.N[2])
                            norm_constant *= self.N(self.cgbf_b.alpha[0], self.cgbf_b.N[0], self.cgbf_b.N[1], self.cgbf_b.N[2])
                            norm_constant *= self.N(self.cgbf_c.alpha[0], self.cgbf_c.N[0], self.cgbf_c.N[1], self.cgbf_c.N[2])
                            norm_constant *= self.N(self.cgbf_d.alpha[0], self.cgbf_d.N[0], self.cgbf_d.N[1], self.cgbf_d.N[2])
                            norm_constant /= self.N(self.cgbf_a.alpha[0], a_lambda_n, 0, 0) # (19)
                            norm_constant /= self.N(self.cgbf_b.alpha[0], b_lambda_n, 0, 0)
                            norm_constant /= self.N(self.cgbf_c.alpha[0], c_lambda_n, 0, 0)
                            norm_constant /= self.N(self.cgbf_d.alpha[0], d_lambda_n, 0, 0)
                            ijkl = ((s*cgbf_b_nfunc+p)*cgbf_c_nfunc+q)*cgbf_d_nfunc+r
                            print "ijkl", ijkl, build_eri[a_lambda_n][b_lambda_n][c_lambda_n][d_lambda_n](&self.libint_data, self.max_num_prim_comb)[ijkl]"""
        return result

    def __dealloc__(self):
        PyMem_Free(self.cgbf_a.alpha)
        PyMem_Free(self.cgbf_b.alpha)
        PyMem_Free(self.cgbf_c.alpha)
        PyMem_Free(self.cgbf_d.alpha)
        PyMem_Free(self.cgbf_a.coef)
        PyMem_Free(self.cgbf_b.coef)
        PyMem_Free(self.cgbf_c.coef)
        PyMem_Free(self.cgbf_d.coef)
        free_libint(&self.libint_data)
