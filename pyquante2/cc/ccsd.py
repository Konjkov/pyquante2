# Algorithms were taken directly from https://github.com/dgasmith/psi4numpy/blob/master/Coupled-Cluster/CCSD.dat
# Special thanks to Daniel Crawford's programming website:
# http://sirius.chem.vt.edu/wiki/doku.php?id=crawdad:programming

import time
import numpy as np
from itertools import product
from functools import reduce

def ccsd(ints, orbs, orbe, ndocc, nvirt, H, verbose=False):
    nmo = ndocc + nvirt
    C = orbs
    eps = orbe

    #Make spin-orbital MO
    t=time.time()
    print 'Starting AO -> spin-orbital MO transformation...'
    nso = nmo * 2
    MO = np.einsum('rJ, pqrs->pqJs', C, ints._2e_ints)
    MO = np.einsum('pI, pqJs->IqJs', C, MO)
    MO = np.einsum('sB, IqJs->IqJB', C, MO)
    MO = np.einsum('qA, IqJB->IAJB', C, MO)

    # Tile MO array so that we have alternating alpha/beta spin orbitals
    MO = np.repeat(MO, 2, axis=0)
    MO = np.repeat(MO, 2, axis=1)
    MO = np.repeat(MO, 2, axis=2)
    MO = np.repeat(MO, 2, axis=3)
    # Build spin mask
    spin_ind = np.arange(nso, dtype=np.int) % 2
    spin_mask = (spin_ind.reshape(-1, 1, 1, 1) == spin_ind.reshape(-1, 1, 1))
    spin_mask = spin_mask * (spin_ind.reshape(-1, 1) == spin_ind)
    # Compute antisymmetrized MO integrals
    MO *= spin_mask
    MO = MO - MO.swapaxes(1, 3)
    MO = MO.swapaxes(1, 2)
    print '..finished transformation in %.3f seconds.\n' % (time.time()-t)
    # Update nocc and nvirt
    nocc = ndocc * 2
    nvirt = MO.shape[0] - nocc
    # Make slices
    o = slice(0, nocc)
    v = slice(nocc, MO.shape[0])
    #Extend eigenvalues
    eps = np.repeat(eps, 2)
    Eocc = eps[o]
    Evirt = eps[v]
    # DPD approach to CCSD equations
    # See: http://sirius.chem.vt.edu/wiki/doku.php?id=crawdad:programming
    # occ orbitals i, j, k, l, m, n
    # virt orbitals a, b, c, d, e, f
    # all oribitals p, q, r, s, t, u, v

    #Bulid Eqn 9: tilde{\Tau})
    def build_tilde_tau(t1, t2):
        ttau = t2.copy()
        tmp = 0.5 * np.einsum('ia, jb->ijab', t1, t1)
        ttau += tmp
        ttau -= tmp.swapaxes(2, 3)
        return ttau

    #Build Eqn 10: \Tau)
    def build_tau(t1, t2):
        ttau = t2.copy()
        tmp = np.einsum('ia,jb->ijab', t1, t1)
        ttau += tmp
        ttau -= tmp.swapaxes(2, 3)
        return ttau

    #Build Eqn 3:
    def build_Fae(t1, t2):
        Fae = F[v, v].copy()
        Fae[np.diag_indices_from(Fae)] = 0
        Fae -= 0.5 * np.einsum('me,ma->ae', F[o, v], t1)
        Fae += np.einsum('mf,mafe->ae', t1, MO[o, v, v, v])
        tmp_tau = build_tilde_tau(t1, t2)
        Fae -= 0.5 * np.einsum('mnaf,mnef->ae', tmp_tau, MO[o, o, v, v])
        return Fae

    #Build Eqn 4:
    def build_Fmi(t1, t2):
        Fmi = F[o, o].copy()
        Fmi[np.diag_indices_from(Fmi)] = 0
        Fmi += 0.5 * np.einsum('ie,me->mi', t1, F[o, v])
        Fmi += np.einsum('ne,mnie->mi', t1, MO[o, o, o, v])
        tmp_tau = build_tilde_tau(t1, t2)
        Fmi += 0.5 * np.einsum('inef,mnef->mi', tmp_tau, MO[o, o, v, v])
        return Fmi

    #Build Eqn 5:
    def build_Fme(t1, t2):
        Fme = F[o, v].copy()
        Fme += np.einsum('nf,mnef->me', t1, MO[o, o, v, v])
        return Fme

    #Build Eqn 6:
    def build_Wmnij(t1, t2):
        Wmnij = MO[o, o, o, o].copy()
        Pij = np.einsum('je,mnie->mnij', t1, MO[o, o, o, v])
        Wmnij += Pij
        Wmnij -= Pij.swapaxes(2, 3)
        tmp_tau = build_tau(t1, t2)
        Wmnij += 0.25 * np.einsum('ijef,mnef->mnij', tmp_tau, MO[o, o, v, v])
        return Wmnij

    #Build Eqn 7:
    def build_Wabef(t1, t2):
        Wabef = MO[v, v, v, v].copy()
        Pab = np.einsum('mb,amef->abef', t1, MO[v, o, v, v])
        Wabef -= Pab
        Wabef += Pab.swapaxes(0, 1)
        tmp_tau = build_tau(t1, t2)
        Wabef += 0.25 * np.einsum('mnab,mnef->abef', tmp_tau, MO[o, o, v, v])
        return Wabef

    #Build Eqn 8:
    def build_Wmbej(t1, t2):
        Wmbej = MO[o, v, v, o].copy()
        Wmbej += np.einsum('jf,mbef->mbej', t1, MO[o, v, v, v])
        Wmbej -= np.einsum('nb,mnej->mbej', t1, MO[o, o, v, o])
        tmp = (0.5 * t2) + np.einsum('jf,nb->jnfb', t1, t1)
        Wmbej -= np.einsum('jnfb,mnef->mbej', tmp, MO[o, o, v, v])
        return Wmbej

    ### Build so Fock matirx
    # Update H, transform to MO basis and tile for alpha/beta spin
    H = np.einsum('uj,vi,uv', C, C, H)
    H = np.repeat(H, 2, axis=0)
    H = np.repeat(H, 2, axis=1)
    # Make H block diagonal
    spin_ind = np.arange(H.shape[0], dtype=np.int) % 2
    H *= (spin_ind.reshape(-1, 1) == spin_ind)
    # Compute Fock matrix
    F = H + np.einsum('pmqm->pq', MO[:, o, :, o])
    ### Build D matrices
    Focc = F[np.arange(nocc), np.arange(nocc)].flatten()
    Fvirt = F[np.arange(nocc, nvirt + nocc), np.arange(nocc, nvirt + nocc)].flatten()
    Dia = Focc.reshape(-1, 1) - Fvirt
    Dijab = Focc.reshape(-1, 1, 1, 1) + Focc.reshape(-1, 1, 1) - Fvirt.reshape(-1, 1) - Fvirt
    ### Construct initial guess
    # t^a_i
    t1 = np.zeros((nocc, nvirt))
    # t^{ab}_{ij}
    MOijab = MO[o, o, v, v]
    t2 = MOijab / Dijab
    ### Compute MP2 in MO basis set to make sure the transformation was correct
    MP2corr_E = np.einsum('ijab,ijab->', MOijab, t2)/4
    #MP2_E = SCF_E + MP2corr_E
    print 'MO based MP2 correlation energy: %.8f' % MP2corr_E
    #print 'MP2 total energy: %.8f' % MP2_E
    #compare_values(energy('mp2'), MP2_E, 6, 'MP2 Energy')
    ### Start Iterations
    E_conv = 1.e-12
    maxiter = 60
    CCSDcorr_E_old = 0.0
    for CCSD_iter in range(1, maxiter + 1):
        ### Build intermediates
        Fae = build_Fae(t1, t2)
        Fmi = build_Fmi(t1, t2)
        Fme = build_Fme(t1, t2)
        Wmnij = build_Wmnij(t1, t2)
        Wabef = build_Wabef(t1, t2)
        Wmbej = build_Wmbej(t1, t2)
        #### Build RHS side of t1 equations
        rhs_T1 = F[o, v].copy()
        rhs_T1 += np.einsum('ie,ae->ia', t1, Fae)
        rhs_T1 -= np.einsum('ma,mi->ia', t1, Fmi)
        rhs_T1 += np.einsum('imae,me->ia', t2, Fme)
        rhs_T1 -= np.einsum('nf,naif->ia', t1, MO[o, v, o, v])
        rhs_T1 -= 0.5 * np.einsum('imef,maef->ia', t2, MO[o, v, v, v])
        rhs_T1 -= 0.5 * np.einsum('mnae,nmei->ia', t2, MO[o, o, v, o])
        ### Build RHS side of t2 equations
        rhs_T2 = MO[o, o, v, v].copy()
        # P_(ab) t_ijae (F_be - 0.5 t_mb F_me)
        tmp = Fae - 0.5 * np.einsum('mb,me->be', t1, Fme)
        Pab = np.einsum('ijae,be->ijab', t2, tmp)
        rhs_T2 += Pab
        rhs_T2 -= Pab.swapaxes(2, 3)
        # P_(ij) t_imab (F_mj + 0.5 t_je F_me)
        tmp = Fmi + 0.5 * np.einsum('je,me->mj', t1, Fme)
        Pij = np.einsum('imab,mj->ijab', t2, tmp)
        rhs_T2 -= Pij
        rhs_T2 += Pij.swapaxes(0, 1)
        tmp_tau = build_tau(t1, t2)
        rhs_T2 += 0.5 * np.einsum('mnab,mnij->ijab', tmp_tau, Wmnij)
        rhs_T2 += 0.5 * np.einsum('ijef,abef->ijab', tmp_tau, Wabef)
        # P_(ij) * P_(ab)
        # (ij - ji) * (ab - ba)
        # ijab - ijba -jiab + jiba
        tmp = np.einsum('ie,ma,mbej->ijab', t1, t1, MO[o, v, v, o])
        Pijab = np.einsum('imae,mbej->ijab', t2, Wmbej)
        Pijab -= tmp
        rhs_T2 += Pijab
        rhs_T2 -= Pijab.swapaxes(2, 3)
        rhs_T2 -= Pijab.swapaxes(0, 1)
        rhs_T2 += Pijab.swapaxes(0, 1).swapaxes(2, 3)
        Pij = np.einsum('ie,abej->ijab', t1, MO[v, v, v, o])
        rhs_T2 += Pij
        rhs_T2 -= Pij.swapaxes(0, 1)
        Pab = np.einsum('ma,mbij->ijab', t1, MO[o, v, o, o])
        rhs_T2 -= Pab
        rhs_T2 += Pab.swapaxes(2, 3)
        ### Update t1 and t2 amplitudes
        t1 = rhs_T1 / Dia
        t2 = rhs_T2 / Dijab
        ### Compute CCSD correlation energy
        CCSDcorr_E = np.einsum('ia,ia->', F[o, v], t1)
        CCSDcorr_E += 0.25 * np.einsum('ijab,ijab->', MO[o, o, v, v], t2)
        CCSDcorr_E += 0.5 * np.einsum('ijab,ia,jb->', MO[o, o, v, v], t1, t1)
        ### Print CCSD correlation energy
        print 'CCSD Iteration %3d: CCSD correlation = %.12f dE = %.5E' % (CCSD_iter, CCSDcorr_E, (CCSDcorr_E - CCSDcorr_E_old))
        if (abs(CCSDcorr_E - CCSDcorr_E_old) < E_conv):
            break
        CCSDcorr_E_old = CCSDcorr_E
    return CCSDcorr_E

if __name__ == '__main__':
    pass
