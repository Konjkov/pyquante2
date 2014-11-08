import numpy as np


def mp2(hamiltonian, orbs, orbe, nocc, nvirt, verbose=False):
    ints = hamiltonian.i2
    moints = ints.transform_mp2(orbs, nocc)
    Evirt, Eocc = orbe[nocc:], orbe[:nocc]

    denominator = 1/(Eocc.reshape(-1, 1, 1, 1) - Evirt.reshape(1, -1, 1, 1) +
                     Eocc.reshape(1, 1, -1, 1) - Evirt.reshape(1, 1, 1, -1))

    MP2corr_OS = np.einsum('iajb,iajb,iajb->', moints, moints, denominator)
    MP2corr_SS = np.einsum('iajb,iajb,iajb->', moints - moints.swapaxes(1,3), moints, denominator)
    return MP2corr_OS + MP2corr_SS


if __name__ == '__main__': test_mp2()
