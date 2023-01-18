import numpy as np
import pytest

from tencirchem import set_backend, PUCCD, UCC
from tencirchem.molecule import h4, h5p
from tests.static.test_static import get_random_integral_and_fci


def test_get_circuit():
    puccd = PUCCD(h5p)
    params = np.random.rand(puccd.n_params)
    s1 = puccd.get_circuit(params).state()
    s2 = puccd.get_circuit(params, trotter=True).state()
    np.testing.assert_allclose(s2, s1, atol=1e-10)
    s3 = puccd.get_circuit(params, givens_swap=True).state()
    np.testing.assert_allclose(s3, s1, atol=1e-10)


@pytest.mark.parametrize("hamiltonian", ["H4", "random integral"])
@pytest.mark.parametrize("engine", ["tensornetwork", "statevector", "civector", "civector-large"])
def test_puccd(hamiltonian, engine, reset_backend):
    if engine in ["tensornetwork", "statevector"]:
        set_backend("jax")

    if hamiltonian == "H4":
        puccd = PUCCD(h4, engine=engine)
        ucc = UCC(h4)
    else:
        nao = 4
        n_elec = 4
        int1e, int2e, _ = get_random_integral_and_fci(nao)
        puccd = PUCCD.from_integral(int1e, int2e, n_elec, engine=engine)
        ucc = UCC.from_integral(int1e, int2e, n_elec)

    # note the order
    # b^\dagger_i = a^\dagger_(nao + i) a^\dagger_(i)
    # b_j = a_(j) a_(nao + j)
    ucc.ex_ops = (
        (7, 3, 0, 4),
        (6, 2, 0, 4),
        (7, 3, 1, 5),
        (6, 2, 1, 5),
    )

    e1 = puccd.kernel()
    ucc.params = puccd.params
    e2 = ucc.energy()

    np.testing.assert_allclose(e1, e2, atol=1e-6)

    rdm1_puccd = puccd.make_rdm1()
    rdm1_ucc = ucc.make_rdm1()
    # slight difference in optimization minimum
    np.testing.assert_allclose(rdm1_puccd, rdm1_ucc, atol=1e-6)

    rdm2_puccd = puccd.make_rdm2(basis="MO")
    rdm2_ucc = ucc.make_rdm2(basis="MO")
    np.testing.assert_allclose(rdm2_puccd, rdm2_ucc, atol=1e-5)
