from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.math cimport exp, pow, sqrt, M_PI
from clibint cimport *


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
        while current>1.0e-9:
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
        int lambda_n
        double A[3]
        int N[3]
        double *alpha
        double *coef

    def __cinit__(self, cgbf):
        for i in range(3):
            self.A[i] = cgbf.origin[i]
            self.N[i] = cgbf.powers[i]
        self.lambda_n = self.N[0] + self.N[1] + self.N[2]
        self.alpha_len = len(cgbf.pexps)
        self.alpha = <double *> PyMem_Malloc(self.alpha_len * sizeof(double))
        self.coef = <double *> PyMem_Malloc(self.alpha_len * sizeof(double))
        for i in range(self.alpha_len):
            self.alpha[i] = cgbf.pexps[i]
            self.coef[i] = cgbf.coefs[i]

    def __dealloc__(self):
        PyMem_Free(self.alpha)
        PyMem_Free(self.coef)


cdef class Libint:
    cdef:
        Libint_t libint_data
        int memory_allocated, memory_required, max_num_prim_comb, max_am
        CGBF a, b, c, d

    def __cinit__(self, CGBF cgbf_a, CGBF cgbf_b, CGBF cgbf_c, CGBF cgbf_d):
        self.a = cgbf_a
        self.b = cgbf_b
        self.c = cgbf_c
        self.d = cgbf_d

        init_libint_base()
        self.max_num_prim_comb = self.a.alpha_len * self.b.alpha_len * self.c.alpha_len * self.d.alpha_len
        self.max_am = max(self.a.lambda_n, self.b.lambda_n, self.c.lambda_n, self.d.lambda_n)
        memory_required = libint_storage_required(self.max_am, self.max_num_prim_comb)
        memory_allocated = init_libint(&self.libint_data, self.max_am, self.max_num_prim_comb)
        for i in xrange(3):
            self.libint_data.AB[i] = self.a.A[i] - self.b.A[i]
            self.libint_data.CD[i] = self.c.A[i] - self.d.A[i]
        primitive_number = 0
        for i in range(self.a.alpha_len):
            for j in range(self.b.alpha_len):
                for k in range(self.c.alpha_len):
                    for l in range(self.d.alpha_len):
                        self.libint_data.PrimQuartet[primitive_number] = self.compute_primitive_data(i, j, k, l)
                        primitive_number += 1

    cdef double N(self, double alpha, int l, int m , int n): # normalization_constant (2)
        cdef double nominator, denominator
        numerator = pow(2, 2 * (l + m + n) + 1.5) * pow(alpha, l + m + n + 1.5)
        denominator = fact2(2 * l - 1) * fact2(2 * m - 1) * fact2(2 * n - 1) * pow(M_PI, 1.5)
        return sqrt(numerator/denominator)

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

        zeta = self.a.alpha[i] + self.b.alpha[j] # (7)
        eta = self.c.alpha[k] + self.d.alpha[l]  # (8)
        rho = zeta * eta / (zeta + eta)          # (9)

        for m in range(3):
            A[m] = self.a.A[m]
            B[m] = self.b.A[m]
            C[m] = self.c.A[m]
            D[m] = self.d.A[m]
            P[m] = (A[m] * self.a.alpha[i] + B[m] * self.b.alpha[j]) / zeta # (10)
            Q[m] = (C[m] * self.c.alpha[k] + D[m] * self.d.alpha[l]) / eta  # (11)
            W[m] = (P[m] * zeta + Q[m] * eta) / (zeta + eta)                # (12)
            pdata.U[0][m] = P[m] - A[m]
            #pdata.U[1][m] = P[m] - B[m] # libderiv
            #pdata.U[1][m] = Q[m] - A[m] # lib12
            pdata.U[2][m] = Q[m] - C[m]
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

        Ca = self.a.coef[i] * self.N(self.a.alpha[i], self.a.lambda_n, 0, 0) # (4)
        Cb = self.b.coef[j] * self.N(self.b.alpha[j], self.b.lambda_n, 0, 0) # (4)
        Cc = self.c.coef[k] * self.N(self.c.alpha[k], self.c.lambda_n, 0, 0) # (4)
        Cd = self.d.coef[l] * self.N(self.d.alpha[l], self.d.lambda_n, 0, 0) # (4)

        S12 = pow(M_PI / zeta, 1.5) * exp(- self.a.alpha[i] * self.b.alpha[j] / zeta * vec_dist2(A, B)) # (15)
        S34 = pow(M_PI / eta, 1.5) * exp(- self.c.alpha[k] * self.d.alpha[l] / eta * vec_dist2(C, D))   # (16)
        m = self.a.lambda_n + self.b.lambda_n + self.c.lambda_n + self.d.lambda_n
        for i in range(m+1):
            pdata.F[i] = 2 * Fm(i, rho * vec_dist2(P, Q)) * sqrt(rho / M_PI) * S12 * S34 * Ca * Cb * Cc * Cd # (17)
        return pdata


    def build_eri(self):
        cdef int n, ijkl
        cdef int cgbf_a_nfunc, cgbf_b_nfunc, cgbf_c_nfunc, cgbf_d_nfunc
        cdef int s, p, q, r
        cdef double norm_constant, result
        if self.max_am==0:
            result = 0.0
            for n in range(self.max_num_prim_comb):
                result += self.libint_data.PrimQuartet[n].F[0]
        else:
            cgbf_a_nfunc = (self.a.lambda_n+1)*(self.a.lambda_n+2)/2
            cgbf_b_nfunc = (self.b.lambda_n+1)*(self.b.lambda_n+2)/2
            cgbf_c_nfunc = (self.c.lambda_n+1)*(self.c.lambda_n+2)/2
            cgbf_d_nfunc = (self.d.lambda_n+1)*(self.d.lambda_n+2)/2
            s = ang_mom_index(self.a.N[0], self.a.N[1], self.a.N[2], self.a.lambda_n)
            p = ang_mom_index(self.b.N[0], self.b.N[1], self.b.N[2], self.b.lambda_n)
            q = ang_mom_index(self.c.N[0], self.c.N[1], self.c.N[2], self.c.lambda_n)
            r = ang_mom_index(self.d.N[0], self.d.N[1], self.d.N[2], self.d.lambda_n)
            norm_constant = 1
            norm_constant *= self.N(self.a.alpha[0], self.a.N[0], self.a.N[1], self.a.N[2])
            norm_constant *= self.N(self.b.alpha[0], self.b.N[0], self.b.N[1], self.b.N[2])
            norm_constant *= self.N(self.c.alpha[0], self.c.N[0], self.c.N[1], self.c.N[2])
            norm_constant *= self.N(self.d.alpha[0], self.d.N[0], self.d.N[1], self.d.N[2])
            norm_constant /= self.N(self.a.alpha[0], self.a.lambda_n, 0, 0) # (19)
            norm_constant /= self.N(self.b.alpha[0], self.b.lambda_n, 0, 0)
            norm_constant /= self.N(self.c.alpha[0], self.c.lambda_n, 0, 0)
            norm_constant /= self.N(self.d.alpha[0], self.d.lambda_n, 0, 0)
            ijkl = ((s*cgbf_b_nfunc+p)*cgbf_c_nfunc+q)*cgbf_d_nfunc+r
            result = build_eri[self.a.lambda_n][self.b.lambda_n][self.c.lambda_n][self.d.lambda_n](&self.libint_data, self.max_num_prim_comb)[ijkl] * norm_constant
        return result

    cdef double *build_shell(self):
        cdef int n, ijkl
        cdef int cgbf_a_nfunc, cgbf_b_nfunc, cgbf_c_nfunc, cgbf_d_nfunc
        cdef int s, p, q, r
        cdef double norm_constant
        cdef double result[1]

        if self.max_am==0:
            result[0] = 0.0
            for n in range(self.max_num_prim_comb):
                result[0] += self.libint_data.PrimQuartet[n].F[0]
            return result
        elif self.max_am==1:
            cgbf_b_nfunc = (self.b.lambda_n+1)*(self.b.lambda_n+2)/2
            cgbf_c_nfunc = (self.c.lambda_n+1)*(self.c.lambda_n+2)/2
            cgbf_d_nfunc = (self.d.lambda_n+1)*(self.d.lambda_n+2)/2
            return build_eri[self.a.lambda_n][self.b.lambda_n][self.c.lambda_n][self.d.lambda_n](&self.libint_data, self.max_num_prim_comb)
        else:
            cgbf_a_nfunc = (self.a.lambda_n+1)*(self.a.lambda_n+2)/2
            cgbf_b_nfunc = (self.b.lambda_n+1)*(self.b.lambda_n+2)/2
            cgbf_c_nfunc = (self.c.lambda_n+1)*(self.c.lambda_n+2)/2
            cgbf_d_nfunc = (self.d.lambda_n+1)*(self.d.lambda_n+2)/2
            for s in range(cgbf_a_nfunc):
                for p in range(cgbf_b_nfunc):
                    for q in range(cgbf_c_nfunc):
                        for r in range(cgbf_d_nfunc):
                            ijkl = ((s*cgbf_b_nfunc+p)*cgbf_c_nfunc+q)*cgbf_d_nfunc+r
                            norm_constant = 1
                            norm_constant *= self.N(self.a.alpha[0], self.a.N[0], self.a.N[1], self.a.N[2])
                            norm_constant *= self.N(self.b.alpha[0], self.b.N[0], self.b.N[1], self.b.N[2])
                            norm_constant *= self.N(self.c.alpha[0], self.c.N[0], self.c.N[1], self.c.N[2])
                            norm_constant *= self.N(self.d.alpha[0], self.d.N[0], self.d.N[1], self.d.N[2])
                            norm_constant /= self.N(self.a.alpha[0], self.a.lambda_n, 0, 0) # (19)
                            norm_constant /= self.N(self.b.alpha[0], self.b.lambda_n, 0, 0)
                            norm_constant /= self.N(self.c.alpha[0], self.c.lambda_n, 0, 0)
                            norm_constant /= self.N(self.d.alpha[0], self.d.lambda_n, 0, 0)
                            result[ijkl] = build_eri[self.a.lambda_n][self.b.lambda_n][self.c.lambda_n][self.d.lambda_n](&self.libint_data, self.max_num_prim_comb)[ijkl] * norm_constant
            return result

    def __dealloc__(self):
        free_libint(&self.libint_data)


def ERI(cgbf_a, cgbf_b, cgbf_c, cgbf_d):

    if cgbf_b.powers[0] + cgbf_b.powers[1] + cgbf_b.powers[2] > cgbf_a.powers[0] + cgbf_a.powers[1] + cgbf_a.powers[2]:
        cgbf_a, cgbf_b = cgbf_b, cgbf_a
    if cgbf_d.powers[0] + cgbf_d.powers[1] + cgbf_d.powers[2] > cgbf_c.powers[0] + cgbf_c.powers[1] + cgbf_c.powers[2]:
        cgbf_c, cgbf_d = cgbf_d, cgbf_c
    if cgbf_a.powers[0] + cgbf_a.powers[1] + cgbf_a.powers[2] + cgbf_b.powers[0] + cgbf_b.powers[1] + cgbf_b.powers[2] > \
       cgbf_c.powers[0] + cgbf_c.powers[1] + cgbf_c.powers[2] + cgbf_d.powers[0] + cgbf_d.powers[1] + cgbf_d.powers[2]:
        cgbf_a, cgbf_b, cgbf_c, cgbf_d = cgbf_c, cgbf_d, cgbf_a, cgbf_b
    return Libint(CGBF(cgbf_a), CGBF(cgbf_b), CGBF(cgbf_c), CGBF(cgbf_d)).build_eri()
