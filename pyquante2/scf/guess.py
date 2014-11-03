from pyquante2.utils import dmat, geigh


def core_hamiltonian_rhf(H):
    name = 'Core hamiltonian guess'
    orbe, orbs = H.eigenv(H.h)
    return H.density(orbs)


def core_hamiltonian_uhf(H):
    name = 'Core hamiltonian guess'
    (orbea, orbsa), (orbeb, orbsb) = H.eigenv(H.h, H.h)
    return H.density(orbsa, orbsb)


def atomic_densities(H):
    name = 'Superposition of atomic densities guess'
