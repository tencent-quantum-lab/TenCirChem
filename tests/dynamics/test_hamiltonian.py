import numpy as np
import pytest
from renormalizer import Model, Mpo, Mps, Op, BasisHalfSpin, BasisSineDVR

from tencirchem import get_dense_operator, TimeEvolution, set_backend
from tencirchem.dynamic.model import pyrazine, sbm
from tencirchem.dynamic.transform import qubit_encode_op, qubit_encode_basis, transform_op, get_subspace_idx


def test_transform():
    op = Op("X Y", 0)
    basis = BasisHalfSpin(0)

    op_transformed = transform_op(op, basis)
    assert len(op_transformed) == 1
    assert op_transformed[0] == Op("Z", 0, 1j)

    op = Op("x", 0)
    basis = BasisSineDVR(0, 4, 0, 1)
    op_transformed = transform_op(op, basis, "binary")
    basis_transformed = qubit_encode_basis([basis], "binary")
    mat1 = get_dense_operator([basis], op)
    mat2 = get_dense_operator(basis_transformed, op_transformed)
    np.testing.assert_allclose(mat1, mat2, atol=1e-10)


@pytest.mark.parametrize("module", [pyrazine, sbm])
@pytest.mark.parametrize("nlevels", [2, 3])
@pytest.mark.parametrize("boson_encoding", [None, "unary", "binary", "gray"])
def test_qubit_encoding(module, nlevels, boson_encoding):
    if boson_encoding is None and nlevels == 3:
        pytest.skip()

    if module == pyrazine:
        nmode = 4
        ham_terms = pyrazine.get_ham_terms()
        basis = pyrazine.get_basis(nlevels)
    else:
        assert module == sbm
        epsilon = 0.1
        delta = 0.23
        nmode = 3
        omega_list = np.random.rand(nmode)
        g_list = np.random.rand(nmode)

        ham_terms = sbm.get_ham_terms(epsilon, delta, nmode, omega_list, g_list)
        basis = sbm.get_basis(omega_list, nlevels=nlevels)

    h1 = Mpo(Model(basis, ham_terms)).todense()

    spin_ham_terms, constant = qubit_encode_op(ham_terms, basis, boson_encoding=boson_encoding)
    # sth must be off if there are so many terms
    assert len(spin_ham_terms) < 200
    spin_basis = qubit_encode_basis(basis, boson_encoding)
    spin_ham_terms[-1] = complex(1) * spin_ham_terms[-1]
    h2 = Mpo(Model(spin_basis, spin_ham_terms)).todense().real
    h2 += np.eye(len(h2)) * constant

    subspace_idx = get_subspace_idx(basis, boson_encoding)
    np.testing.assert_allclose(h2[subspace_idx][:, subspace_idx], h1, atol=1e-10)


@pytest.mark.parametrize("boson_encoding", ["unary", "binary", "gray"])
def test_init_circuit(boson_encoding, reset_backend):
    backend = set_backend("jax")
    nlevels = 3
    ham_terms = pyrazine.get_ham_terms()
    basis = pyrazine.get_basis(nlevels)
    init_condition = {"10a": 2, "6a": [np.sqrt(2) / 2, 0, -np.sqrt(2) / 2]}
    tc = TimeEvolution(ham_terms, basis, boson_encoding=boson_encoding, init_condition=init_condition)
    ref_state = Mps.hartree_product_state(Model(basis, ham_terms), condition=init_condition).todense()

    subspace_idx = get_subspace_idx(basis, boson_encoding)

    np.testing.assert_allclose(backend.numpy(tc.state)[subspace_idx], ref_state)
