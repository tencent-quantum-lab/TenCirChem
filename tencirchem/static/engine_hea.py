from functools import partial

import numpy as np
import tensorcircuit as tc
from tensorcircuit import DMCircuit
from tensorcircuit.noisemodel import circuit_with_noise
from tensorcircuit.experimental import parameter_shift_grad

from tencirchem.utils.backend import jit


@partial(jit, static_argnums=[1])
def get_statevector(params, get_circuit):
    return get_circuit(params).state()


@partial(jit, static_argnums=[1, 2])
def get_densitymatrix(params, get_dmcircuit, noise_conf):
    if noise_conf is None:
        raise ValueError("NoiseConf not provided")
    circuit = get_dmcircuit(params)
    circuit = circuit_with_noise(circuit, noise_conf)
    return circuit.densitymatrix()


@partial(jit, static_argnums=[2])
def get_energy_tensornetwork(params, h_array, get_circuit):
    s = get_statevector(params, get_circuit)
    return (s.conj() @ (h_array @ s)).real


@partial(jit, static_argnums=[2, 3])
def get_energy_tensornetwork_noise(params, h_array, get_dmcircuit, noise_conf):
    dm = get_densitymatrix(params, get_dmcircuit, noise_conf)
    return tc.backend.trace(dm @ h_array).real


@partial(jit, static_argnums=[1, 3, 4, 5])
def get_energy_tensornetwork_noise_shot(params, paulis, coeffs, get_dmcircuit, noise_conf, shots: int):
    dm = get_densitymatrix(params, get_dmcircuit, noise_conf)
    n_qubits = round(np.log2(dm.shape[0]))
    c = DMCircuit(n_qubits, dminputs=dm)
    expectations = []
    for pauli, coef in zip(paulis, coeffs):
        x = []
        y = []
        z = []
        m = {"X": x, "Y": y, "Z": z}
        for idx, symbol in pauli:
            m[symbol].append(idx)
        expectation = coef
        if len(x + y + z) != 0:
            expectation *= c.sample_expectation_ps(x=x, y=y, z=z, shots=shots, noise_conf=noise_conf)
        expectations.append(expectation)
    # already real
    return sum(expectations)


def _get_energy_and_grad(partial_get_energy, params, grad):
    if grad == "param-shift":
        e = partial_get_energy(params)
        grad_f = parameter_shift_grad(partial_get_energy)
        g = grad_f(params)
    else:
        assert grad == "autodiff"
        e, g = tc.backend.value_and_grad(partial_get_energy)(params)
    return e, g


@partial(jit, static_argnums=[2, 3])
def get_energy_and_grad_tensornetwork(params, h_array, get_circuit, grad):
    partial_get_energy = partial(get_energy_tensornetwork, h_array=h_array, get_circuit=get_circuit)
    return _get_energy_and_grad(partial_get_energy, params, grad)


@partial(jit, static_argnums=[2, 3, 4])
def get_energy_and_grad_tensornetwork_noise(params, h_array, get_dmcircuit, noise_conf, grad):
    partial_get_energy = partial(
        get_energy_tensornetwork_noise, h_array=h_array, get_dmcircuit=get_dmcircuit, noise_conf=noise_conf
    )
    return _get_energy_and_grad(partial_get_energy, params, grad)


@partial(jit, static_argnums=[1, 3, 4, 5, 6])
def get_energy_and_grad_tensornetwork_noise_shot(params, paulis, coeffs, get_dmcircuit, noise_conf, shots, grad):
    partial_get_energy = partial(
        get_energy_tensornetwork_noise_shot,
        paulis=paulis,
        coeffs=coeffs,
        get_dmcircuit=get_dmcircuit,
        noise_conf=noise_conf,
        shots=shots,
    )
    return _get_energy_and_grad(partial_get_energy, params, grad)
