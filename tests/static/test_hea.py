import numpy as np
import pytest
from qiskit.circuit.library import RealAmplitudes

from tencirchem import UCCSD, HEA, parity, set_backend
from tencirchem.molecule import h2
from tests.static.test_engine import set_backend_with_skip


@pytest.mark.parametrize(
    "engine",
    [
        "tensornetwork",
        "tensornetwork-noise",
        "tensornetwork-shot",
        "tensornetwork-noise&shot",
    ],
)
@pytest.mark.parametrize("backend_str", ["jax", "numpy"])
@pytest.mark.parametrize("grad", ["param-shift", "autodiff", "free"])
def test_hea(engine, backend_str, grad, reset_backend):
    if backend_str in ["numpy", "cupy"] and grad == "autodiff":
        pytest.xfail("Incompatible backend and gradient method")
    if engine in ["tensornetwork-shot", "tensornetwork-noise&shot"] and grad == "autodiff":
        pytest.xfail("Incompatible engine and gradient method")
    set_backend_with_skip(backend_str)
    m = h2
    uccsd = UCCSD(m)
    hea = HEA.ry(uccsd.int1e, uccsd.int2e, uccsd.n_elec, uccsd.e_core, 3, engine=engine)
    hea.grad = grad
    e = hea.kernel()
    atol = 0.1
    if engine == "tensornetwork-noise&shot" and grad == "free":
        atol *= 2
    np.testing.assert_allclose(e, uccsd.e_fci, atol=atol)


def test_qiskit_circuit():
    m = h2
    uccsd = UCCSD(m)
    circuit = RealAmplitudes(2)
    hea = HEA(parity(uccsd.h_fermion_op, 4, 2), circuit, np.random.rand(circuit.num_parameters))
    e = hea.kernel()
    np.testing.assert_allclose(e, uccsd.e_fci, atol=1e-5)
    hea.print_summary()


@pytest.mark.parametrize("mapping", ["jordan-wigner", "bravyi-kitaev"])
def test_mapping(mapping):
    uccsd = UCCSD(h2)
    hea = HEA.ry(uccsd.int1e, uccsd.int2e, uccsd.n_elec, uccsd.e_core, 3, mapping=mapping)
    e = hea.kernel()
    np.testing.assert_allclose(e, uccsd.e_fci)


@pytest.mark.parametrize("mapping", ["jordan-wigner", "parity", "bravyi-kitaev"])
def test_rdm(mapping, reset_backend):
    set_backend("jax")
    uccsd = UCCSD(h2)
    uccsd.kernel()
    hea = HEA.ry(uccsd.int1e, uccsd.int2e, uccsd.n_elec, uccsd.e_core, 3, mapping=mapping)
    hea.kernel()
    np.testing.assert_allclose(hea.make_rdm1(), uccsd.make_rdm1(basis="MO"), atol=1e-4)
    np.testing.assert_allclose(hea.make_rdm2(), uccsd.make_rdm2(basis="MO"), atol=1e-4)
