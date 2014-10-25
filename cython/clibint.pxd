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


cdef extern from "libderiv.h":

    ctypedef struct Libderiv_t:
        prim_data *PrimQuartet
        double *ABCD[12+144]
        double AB[3]
        double CD[3]

    void (*build_deriv1_eri[4][4][4][4])(Libderiv_t *, int)
    void (*build_deriv12_eri[3][3][3][3])(Libderiv_t *, int)
    void init_libderiv_base()

    int  init_libderiv1(Libderiv_t *, int max_am, int max_num_prim_quartets, int max_cart_class_size)
    int  init_libderiv12(Libderiv_t *, int max_am, int max_num_prim_quartets, int max_cart_class_size)
    void free_libderiv(Libderiv_t *)

    int  libderiv1_storage_required(int max_am, int max_num_prim_quartets, int max_cart_class_size)
    int  libderiv12_storage_required(int max_am, int max_num_prim_quartets, int max_cart_class_size)


cdef extern from "libr12.h":

    ctypedef struct contr_data:
        double AB[3]
        double CD[3]
        double AC[3]
        double ABdotAC, CDdotCA

    ctypedef struct Libr12_t:
        prim_data *PrimQuartet
        contr_data ShellQuartet

    void (*build_r12_gr[4][4][4][4])(Libr12_t *, int)
    void (*build_r12_grt[4][4][4][4])(Libr12_t *, int)
    void init_libr12_base()

    int  init_libr12(Libr12_t *, int max_am, int max_num_prim_quartets)
    void free_libr12(Libr12_t *)
    int  libr12_storage_required(int max_am, int max_num_prim_quartets)
