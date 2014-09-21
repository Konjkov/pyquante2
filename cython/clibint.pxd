cdef extern from "libint.h":

    ctypedef struct prim_data:
        double F[17]
        double U[6][3]
        double twozeta_a
        double twozeta_b
        double twozeta_c
        double twozeta_d
        double oo2z
        double oo2n
        double oo2zn
        double poz
        double pon
        double oo2p
        double ss_r12_ss

    ctypedef struct Libint_t:
        prim_data *PrimQuartet
        double AB[3]
        double CD[3]

    double *(*build_eri[5][5][5][5])(Libint_t *, int)
    void init_libint_base()
    int  init_libint(Libint_t *, int max_am, int max_num_prim_comb)
    void free_libint(Libint_t *)
    int  libint_storage_required(int max_am, int max_num_prim_comb)
