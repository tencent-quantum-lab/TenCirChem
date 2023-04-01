import numpy as np
import pytest
from renormalizer import Op

from tencirchem import set_backend
from tencirchem.dynamic import sbm, qubit_encode_op, qubit_encode_basis, TimeEvolution


@pytest.mark.parametrize("algorithm", ["vanilla", "include_phase", "p-VQD", "trotter"])
def test_sbm(reset_backend, algorithm):
    set_backend("jax")
    epsilon = 0
    delta = 1
    nmode = 1
    omega_list = [1]
    g_list = [0.5]
    nbas = 8
    n_layers = 3

    ham_terms = sbm.get_ham_terms(epsilon, delta, nmode, omega_list, g_list)
    basis = sbm.get_basis(omega_list, [nbas] * nmode)
    ham_terms_spin, _ = qubit_encode_op(ham_terms, basis, "gray")
    basis_spin = qubit_encode_basis(basis, "gray")

    te = TimeEvolution(
        ham_terms_spin,
        basis_spin,
        n_layers=n_layers,
        property_op_dict={"Z": Op("Z", "spin"), "X": Op("X", "spin")},
        eps=1e-5,
    )
    te.include_phase = algorithm == "include_phase"

    if algorithm in ["vanilla", "include_phase"]:
        algo = "vqd"
        tau = 0.1
    elif algorithm == "p-VQD":
        algo = "pvqd"
        tau = 0.1
    else:
        algo = "trotter"
        tau = 0.02

    for _ in range(50):
        te.kernel(tau, algo=algo)
    z = te.property_dict["Z"]
    np.testing.assert_allclose(z[:, 0], z[:, 1], atol=1e-2)

    x = te.property_dict["X"]
    np.testing.assert_allclose(x[:, 0], x[:, 1], atol=1e-2)
