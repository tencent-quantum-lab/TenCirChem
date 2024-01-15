#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.


from functools import partial

import numpy as np
import tensorcircuit as tc
from tensorcircuit import Circuit, DMCircuit
from tensorcircuit.noisemodel import circuit_with_noise
from tensorcircuit.experimental import parameter_shift_grad
from tensorcircuit.cloud.wrapper import batch_expectation_ps

from tencirchem.utils.backend import jit


class QpuConf:
    def __init__(self, device=None, provider=None, initial_mapping=None):
        if device is None:
            device = "tianji_s2"
        self.device = device
        self.privider = provider
        self.initial_mapping = initial_mapping


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


def sample_expectation_pauli(c, paulis, coeffs, shots, noise_conf):
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


@partial(jit, static_argnums=[1, 3, 4])
def get_energy_tensornetwork_shot(params, paulis, coeffs, get_circuit, shots):
    c = get_circuit(params)
    return sample_expectation_pauli(c, paulis, coeffs, shots, None)


@partial(jit, static_argnums=[1, 3, 4, 5])
def get_energy_tensornetwork_noise_shot(params, paulis, coeffs, get_dmcircuit, noise_conf, shots: int):
    dm = get_densitymatrix(params, get_dmcircuit, noise_conf)
    n_qubits = round(np.log2(dm.shape[0]))
    c = DMCircuit(n_qubits, dminputs=dm)
    return sample_expectation_pauli(c, paulis, coeffs, shots, noise_conf)


def get_energy_qpu(params, paulis, coeffs, get_circuit, qpu_conf: QpuConf, shots: int):
    c: Circuit = get_circuit(params)
    pss = []
    symbol_mapping = {"X": 1, "Y": 2, "Z": 3}
    ps_identity_idx = []
    for i, pauli in enumerate(paulis):
        ps = [0] * c.circuit_param["nqubits"]
        if len(pauli) == 0:
            ps_identity_idx.append(i)
            continue
        for idx, symbol in pauli:
            ps[idx] = symbol_mapping[symbol]
        pss.append(ps)
    assert len(pss) + len(ps_identity_idx) == len(paulis)
    coeffs_non_identity = [coeffs[i] for i in range(len(coeffs)) if i not in ps_identity_idx]
    assert len(pss) == len(coeffs_non_identity)
    es = []
    for _ in range((shots - 1) // 8192 + 1):
        e = batch_expectation_ps(c, pss, device=qpu_conf.device, ws=coeffs_non_identity, shots=8192)
        es.append(e)
    print(paulis)
    print(coeffs)
    print(es)
    return np.mean(es) + sum([coeffs[i] for i in ps_identity_idx])


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


@partial(jit, static_argnums=[1, 3, 4, 5])
def get_energy_and_grad_tensornetwork_shot(params, paulis, coeffs, get_circuit, shots, grad):
    partial_get_energy = partial(
        get_energy_tensornetwork_shot,
        paulis=paulis,
        coeffs=coeffs,
        get_circuit=get_circuit,
        shots=shots,
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


def get_energy_and_grad_qpu(params, paulis, coeffs, get_circuit, shots: int, grad):
    partial_get_energy = partial(
        get_energy_qpu,
        paulis=paulis,
        coeffs=coeffs,
        get_circuit=get_circuit,
        shots=shots,
    )
    return _get_energy_and_grad(partial_get_energy, params, grad)
