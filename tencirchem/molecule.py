#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.


import numpy as np
from pyscf.gto.mole import Mole
from pyscf import M, ao2mo

from tencirchem.static.hamiltonian import random_integral


# Duck-typing PySCF Mole object. Not supposed to be an external user-interface
class _Molecule(Mole):
    @classmethod
    def random(cls, nao, n_elec, seed=2077):
        int1e, int2e = random_integral(nao, seed)
        return cls(int1e, int2e, n_elec)

    def __init__(self, int1e, int2e, n_elec: int, e_nuc: float = 0, ovlp: np.ndarray = None):
        super().__init__()

        self.nao = len(int1e)
        self.int1e = int1e
        self.int2e = int2e
        self.int2e_s8 = ao2mo.restore(8, self.int2e, self.nao)
        # in PySCF m.nelec returns a tuple and m.nelectron returns an integer
        # So here use n_elec to indicate the difference
        self.n_elec = self.nelectron = n_elec
        self.e_nuc = e_nuc
        if ovlp is None:
            self.ovlp = np.eye(self.nao)
        else:
            self.ovlp = ovlp
        # self.symmetry = True
        # avoid sanity check
        self.verbose = 0
        self.build()
        self.incore_anyway = True

    def intor(self, intor, comp=None, hermi=0, aosym="s1", out=None, shls_slice=None, grids=None):
        if intor == "int1e_kin":
            return np.zeros_like(self.int1e)
        elif intor == "int1e_nuc":
            return self.int1e
        elif intor == "int1e_ovlp":
            return self.ovlp
        elif intor == "int2e":
            assert aosym in ["s8", "s1"]
            if aosym == "s1":
                return self.int2e
            else:
                return self.int2e_s8
        else:
            raise ValueError(f"Unsupported integral type: {intor}")

    intor_symmetric = intor

    def tot_electrons(self):
        return self.n_elec

    def energy_nuc(self):
        return self.e_nuc


# for testing/benchmarking

_random = _Molecule.random


def h_chain(n_h=4, bond_distance=0.8, charge=0):
    return M(atom=[["H", 0, 0, bond_distance * i] for i in range(n_h)], charge=charge)


def h_ring(n_h=4, radius=0.7):
    atoms = []
    for angle in np.linspace(0, 2 * np.pi, n_h + 1)[:-1]:
        atom = ["H", 0, np.cos(angle) * radius, np.sin(angle) * radius]
        atoms.append(atom)
    return M(atom=atoms)


H2 = h2 = h_chain(2, 0.741)

H3p = h3p = h_chain(3, charge=1)

H4 = h4 = h_chain()

H5p = h5p = h_chain(5, charge=1)

H6 = h6 = h_chain(6)

H8 = h8 = h_chain(8)


def water(bond_length=0.9584, bond_angle=104.45, basis="sto3g"):
    bond_angle = bond_angle / 180 * np.pi
    phi = bond_angle / 2
    r = bond_length
    O = ["O", 0, 0, 0]
    H1 = ["H", -r * np.sin(phi), r * np.cos(phi), 0]
    H2 = ["H", r * np.sin(phi), r * np.cos(phi), 0]
    return M(atom=[O, H1, H2], basis=basis)


H2O = h2o = water


HeHp = hehp = lambda d=1: M(atom=[["H", 0, 0, 0], ["He", 0, 0, d]], charge=1)
LiH = lih = lambda d=1.6: M(atom=[["H", 0, 0, 0], ["Li", 0, 0, d]])
BeH2 = beh2 = lambda d=1.6: M(atom=[["H", 0, 0, -d], ["Be", 0, 0, 0], ["H", 0, 0, d]])

N2 = n2 = nitrogen = lambda d=1.09: M(atom=[["N", 0, 0, 0], ["N", 0, 0, d]])


def get_ch4_coord(d):
    x = d / np.sqrt(3)
    ch4_coord = [
        ["C", 0, 0, 0],
        ["H", -x, x, x],
        ["H", x, -x, x],
        ["H", -x, -x, -x],
        ["H", x, x, -x],
    ]
    return ch4_coord


CH4 = ch4 = methane = lambda x=1.09: M(atom=get_ch4_coord(x))


if __name__ == "__main__":
    n = 6
    h1 = np.zeros((n, n))
    for i in range(n - 1):
        h1[i, i + 1] = h1[i + 1, i] = -1.0
    h1[n - 1, 0] = h1[0, n - 1] = -1.0
    eri = np.zeros((n, n, n, n))
    for i in range(n):
        eri[i, i, i, i] = 2.0

    m = _Molecule(h1, eri, 6)
    from pyscf.scf import RHF

    rhf = RHF(m)
    # avoid serialization warning
    rhf.chkfile = False
    print(rhf.kernel())
