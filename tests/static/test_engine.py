import numpy as np
import pytest

from tencirchem import UCCSD, set_backend
from tencirchem.molecule import h4
from tencirchem.static.engine_ucc import apply_excitation
from tencirchem.static.ci_utils import get_init_civector
from tencirchem.static.evolve_tensornetwork import get_init_circuit


@pytest.fixture
def ref_eg():
    uccsd = UCCSD(h4)

    np.random.seed(2077)
    params = np.random.rand(len(uccsd.init_guess)) - 0.5

    set_backend("jax")
    # the simplest method is the most reliable one
    e, g = uccsd.energy_and_grad(params, engine="tensornetwork")
    set_backend("numpy")
    return e, g


@pytest.mark.parametrize("engine", ["statevector", "civector", "civector-large", "pyscf"])
def test_gradient(ref_eg, engine, reset_backend):
    e_ref, g_ref = ref_eg

    uccsd = UCCSD(h4)

    np.random.seed(2077)
    params = np.random.rand(len(uccsd.init_guess)) - 0.5

    if engine == "statevector":
        set_backend("jax")
    else:
        set_backend("numpy")
    e, g = uccsd.energy_and_grad(params, engine=engine)

    np.testing.assert_allclose(e, e_ref, atol=1e-5)
    np.testing.assert_allclose(g, g_ref, atol=1e-5)


@pytest.fixture
def ref_state():
    uccsd = UCCSD(h4)

    np.random.seed(2077)
    params = np.random.rand(len(uccsd.init_guess)) - 0.5
    set_backend("jax")
    engine = "tensornetwork"
    state = uccsd.civector(params, engine=engine)
    state = apply_excitation(state, n_qubits=8, n_elec=4, ex_op=(4, 0, 3, 7), hcb=False, engine=engine)
    set_backend("numpy")
    return state


@pytest.mark.parametrize("engine", ["statevector", "civector", "civector-large", "pyscf"])
def test_excitation(ref_state, engine, reset_backend):
    uccsd = UCCSD(h4)

    np.random.seed(2077)
    params = np.random.rand(len(uccsd.init_guess)) - 0.5

    if engine == "statevector":
        set_backend("jax")
    else:
        set_backend("numpy")
    state = uccsd.civector(params, engine=engine)
    state = apply_excitation(state, n_qubits=8, n_elec=4, ex_op=(4, 0, 3, 7), hcb=False, engine=engine)
    np.testing.assert_allclose(state, ref_state, atol=1e-6)


def set_backend_with_skip(backend_str):
    if backend_str == "jax":
        # Detect GPU, https://github.com/google/jax/issues/971
        from jax.lib import xla_bridge

        if xla_bridge.get_backend().platform == "cpu":
            pytest.skip("skip JAX + CPU because of the slow compilation")
    elif backend_str == "cupy":
        try:
            import cupy
        except ImportError:
            pytest.skip("CuPy is not installed")
    return set_backend(backend_str)


@pytest.mark.parametrize("engine", ["tensornetwork", "statevector", "civector", "civector-large", "pyscf"])
@pytest.mark.parametrize("backend_str", ["jax", "numpy", "cupy"])
@pytest.mark.parametrize("init_state", [None, "civector", "circuit"])
def test_gradient_opt(backend_str, engine, init_state, reset_backend):
    if engine in ["tensornetwork", "statevector"] and backend_str in ["numpy", "cupy"]:
        pytest.xfail("Incompatible engine and backend")
    set_backend_with_skip(backend_str)
    uccsd = UCCSD(h4, engine=engine)
    # test initial condition. Has no effect
    if init_state == "civector":
        uccsd.init_state = get_init_civector(uccsd.civector_size)
    elif init_state == "circuit":
        uccsd.init_state = get_init_circuit(uccsd.n_qubits, uccsd.n_elec, uccsd.hcb)
    e = uccsd.kernel()
    np.testing.assert_allclose(e, uccsd.e_fci, atol=1e-4)
