import numpy as np
from pyscf import fci
from pyscf.scf import RHF

from tencirchem.molecule import _Molecule, h2o
from tencirchem.static.hamiltonian import get_integral_from_hf


def test_molecule():
    mol = h2o()
    mol.verbose = 0

    mf1 = RHF(mol)
    mf1.verbose = 0
    e_hf1 = mf1.kernel()
    e_fci1, _ = fci.FCI(mf1).kernel()

    int1e = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    int2e = mol.intor("int2e")
    n_elec = mol.nelectron
    e_nuc = mol.energy_nuc()
    ovlp = mol.intor("int1e_ovlp")
    mol = _Molecule(int1e, int2e, n_elec, e_nuc, ovlp)
    mf2 = RHF(mol)

    mf2.chkfile = None
    mf2.init_guess = "1e"
    mf2.verbose = 0
    e_hf2 = mf2.kernel()
    e_fci2, _ = fci.FCI(mf2).kernel()

    np.testing.assert_allclose(mf1.get_hcore(), mf2.get_hcore())
    np.testing.assert_allclose(mf1.get_ovlp(), mf2.get_ovlp())
    np.testing.assert_allclose(mf1._eri, mf2._eri, atol=1e-10)
    np.testing.assert_allclose(mf1.energy_nuc(), mf2.energy_nuc())
    np.testing.assert_allclose(mf1.energy_elec(), mf2.energy_elec())
    np.testing.assert_allclose(e_hf1, e_hf2)
    np.testing.assert_allclose(e_fci1, e_fci2)

    int1e, int2e, e_core = get_integral_from_hf(mf2)
    e_fci3, _ = fci.direct_spin1.kernel(int1e, int2e, mol.nao, mol.n_elec)
    np.testing.assert_allclose(e_fci2, e_fci3 + e_core)
