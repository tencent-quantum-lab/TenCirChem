import numpy as np

import pytest
from scipy.sparse import linalg
from pyscf import fci
from openfermion.linalg import eigenspectrum
import tensorcircuit as tc

from tencirchem import UCCSD
from tencirchem.molecule import h4, h6, _random
from tencirchem.static.hamiltonian import get_h_from_hf, mpo_to_quoperator
from tencirchem.static.ci_utils import get_ci_strings
from tencirchem.static.hea import binary, parity


@pytest.mark.parametrize("m", [h4, _random(4, 4)])
@pytest.mark.parametrize("hcb", [False, True])
@pytest.mark.parametrize("htype", ["sparse"])
def test_hamiltonian(m, hcb, htype):
    hf = m.HF()
    hf.chkfile = None
    hf.verbose = 0
    hf.kernel()

    hamiltonian = get_h_from_hf(hf, hcb=hcb, htype=htype)
    if htype == "mpo":
        hamiltonian = mpo_to_quoperator(hamiltonian).eval_matrix()
    else:
        hamiltonian = np.array(hamiltonian.todense())

    e_nuc = hf.energy_nuc()
    if not hcb:
        fci_e, _ = fci.FCI(hf).kernel()
        # not generally true but works for this example
        np.testing.assert_allclose(np.linalg.eigh(hamiltonian)[0][0] + e_nuc, fci_e, atol=1e-6)
    else:
        circuit = tc.Circuit(4)
        for i in range(4 // 2):
            circuit.X(3 - i)
        state = circuit.state()
        np.testing.assert_allclose(state @ (hamiltonian @ state).reshape(-1) + e_nuc, hf.e_tot)


@pytest.mark.parametrize("m", [h4, _random(4, 4)])
@pytest.mark.parametrize("hcb", [False, True])
def test_hamiltonian_fcifunc(m, hcb):
    hf = m.HF()
    hf.chkfile = None
    hf.verbose = 0
    hf.kernel()

    htype = "fcifunc"
    hamiltonian = get_h_from_hf(hf, hcb=hcb, htype=htype)
    hamiltonian_ref = get_h_from_hf(hf, hcb=hcb, htype="sparse").todense()

    if not hcb:
        n_qubits = 8
    else:
        n_qubits = 4
    ci_strings = get_ci_strings(n_qubits, 4, hcb)
    hamiltonian = linalg.LinearOperator(shape=(len(ci_strings), len(ci_strings)), matvec=hamiltonian, dtype=np.float64)
    e = linalg.eigsh(hamiltonian, k=1, which="SA")[0][0]
    e_ref = np.linalg.eigh(hamiltonian_ref)[0][0]
    np.testing.assert_allclose(e, e_ref)


@pytest.mark.parametrize("m", [h4, _random(4, 4), h6])
@pytest.mark.parametrize("encode", [binary, parity])
def test_hea_encoding(m, encode):
    uccsd = UCCSD(m)
    qubit_hamiltonian = encode(uccsd.h_fermion_op, uccsd.n_qubits, uccsd.n_elec)
    e1 = eigenspectrum(qubit_hamiltonian)[0]
    np.testing.assert_allclose(e1, uccsd.e_fci)
