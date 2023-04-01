#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.


import logging
from typing import Any, List, Tuple
from functools import partial

import numpy as np
from openfermion import jordan_wigner
import tensorcircuit as tc

from tencirchem.static.ci_utils import get_ci_strings, civector_to_statevector
from tencirchem.utils.backend import jit
from tencirchem.utils.misc import ex_op_to_fop, reverse_qop_idx
from tencirchem.utils.circuit import evolve_pauli, multicontrol_ry

logger = logging.getLogger(__name__)


Tensor = Any


def evolve_excitation(circuit: tc.Circuit, f_idx: tuple, qop, theta, hcb: bool, decompose_multicontrol: bool):
    n_qubits = circuit.circuit_param["nqubits"]
    f_idx = [n_qubits - 1 - idx for idx in f_idx]

    z_indices = []
    if not hcb:
        for idx, term in next(iter(qop.terms.keys())):
            if term != "Z":
                assert idx in f_idx
                continue
            z_indices.append(idx)

    def parity_gates(target, reverse=False):
        if len(z_indices) == 0:
            return
        if not reverse:
            for ii in range(len(z_indices) - 1):
                circuit.CNOT(z_indices[ii], z_indices[ii + 1])
            circuit.cz(z_indices[-1], target)
        else:
            circuit.cz(z_indices[-1], target)
            for ii in reversed(range(len(z_indices) - 1)):
                circuit.CNOT(z_indices[ii], z_indices[ii + 1])

    if len(f_idx) == 2:
        k, l = f_idx
        circuit.CNOT(k, l)
        parity_gates(k)
        circuit.cry(l, k, theta=theta)
        parity_gates(k, reverse=True)
        circuit.CNOT(k, l)
    else:
        assert len(f_idx) == 4
        k, l, i, j = f_idx
        circuit.CNOT(l, k)
        circuit.CNOT(j, i)
        circuit.CNOT(l, j)
        parity_gates(l)
        if not decompose_multicontrol:
            try:
                name = f"Ry({theta:.4f})"
            except TypeError:
                # jax tracer can't be formatted
                name = "Ry"
            circuit.multicontrol(i, j, k, l, ctrl=[0, 1, 0], unitary=tc.gates.ry_gate(theta).tensor, name=name)
        else:
            circuit.append(multicontrol_ry(theta), indices=[i, j, k, l])
        parity_gates(l, reverse=True)
        circuit.CNOT(l, j)
        circuit.CNOT(j, i)
        circuit.CNOT(l, k)

    return circuit


def get_init_circuit(n_qubits, n_elec, hcb, init_state=None, givens_swap=False):
    if init_state is None:
        # prepare HF state
        circuit = tc.Circuit(n_qubits)
        if not hcb:
            for i in range(n_elec // 2):
                circuit.X(n_qubits - 1 - i)
                circuit.X(n_qubits // 2 - 1 - i)
        else:
            if not givens_swap:
                for i in range(n_elec // 2):
                    circuit.X(n_qubits - 1 - i)
            else:
                for i in range(n_elec // 2):
                    circuit.X(i)
    elif isinstance(init_state, tc.Circuit):
        return init_state
    else:
        ci_strings = get_ci_strings(n_qubits, n_elec, hcb)
        statevector = civector_to_statevector(init_state, n_qubits, ci_strings)
        if givens_swap:
            statevector = statevector.reshape([2] * n_qubits)
            new_idx = list(range(n_qubits - n_elec // 2, n_qubits)) + list(range(n_qubits - n_elec // 2))
            statevector = statevector.transpose(new_idx).ravel()
        circuit = tc.Circuit(n_qubits, inputs=statevector)
    return circuit


def get_circuit(
    params,
    n_qubits,
    n_elec,
    ex_ops,
    param_ids,
    hcb: bool = False,
    init_state: tc.Circuit = None,
    decompose_multicontrol: bool = False,
    trotter: bool = False,
):
    if param_ids is None:
        assert len(params) == len(ex_ops)
        param_ids = list(range(len(params)))

    circuit = get_init_circuit(n_qubits, n_elec, hcb, init_state)

    for param_id, f_idx in zip(param_ids, ex_ops):
        theta = params[param_id]
        fop = ex_op_to_fop(f_idx, with_conjugation=True)
        qop = reverse_qop_idx(jordan_wigner(fop), n_qubits)
        if trotter:
            for pauli_string, v in qop.terms.items():
                if hcb:
                    pauli_string = [(idx, symbol) for idx, symbol in pauli_string if symbol != "Z"]
                circuit = evolve_pauli(circuit, pauli_string, -2 * v.imag * theta)
        else:
            # https://arxiv.org/pdf/2005.14475.pdf
            circuit = evolve_excitation(circuit, f_idx, qop, 2 * theta, hcb, decompose_multicontrol)

    return circuit


def get_gs_unitary(theta):
    # SWAP @ Givens Rotation
    a = [
        [1, 0, 0, 0],
        [0, -tc.backend.sin(theta), tc.backend.cos(theta), 0],
        [0, tc.backend.cos(theta), tc.backend.sin(theta), 0],
        [0, 0, 0, 1],
    ]
    return tc.backend.convert_to_tensor(a)


def get_gs_indices(no: int, nv: int) -> List[Tuple[int, int]]:
    """Givens-Swap indices"""
    layer1 = [np.array([no - 1, no])]
    for _ in range(nv - 1):
        layer1.append(layer1[-1] + 1)
    layer1 = np.array(layer1)
    ret = [layer1]
    for _ in range(no - 1):
        ret.append(ret[-1] - 1)
    ret = np.array(ret).reshape(-1, 2)
    assert len(ret) == no * nv
    return ret.tolist()


def get_circuit_givens_swap(params, n_qubits, n_elec, init_state=None):
    # https://arxiv.org/pdf/2002.00035.pdf
    # the swapped qubit index represents molecule orbitals 0 to n_qubits - 1
    # so we need to take negative of theta
    #             ┌───┐        ┌──────┐
    # q_0(MO 1) : ┤ X ├────────┤      ├─────────  MO 3
    #             ├───┤┌──────┐│  GS  │┌──────┐
    # q_1(MO 0) : ┤ X ├┤      ├┤ 3, 1 ├┤      ├─  MO 2
    #             └───┘│  GS  │├──────┤│  GS  │
    # q_2(MO 3) : ─────┤ 3, 0 ├┤      ├┤ 2, 1 ├─  MO 1
    #                  └──────┘│  GS  │└──────┘
    # q_3(MO 2) : ─────────────┤ 2, 0 ├─────────  MO 0
    #                          └──────┘
    circuit = get_init_circuit(n_qubits, n_elec, hcb=True, init_state=init_state, givens_swap=True)
    gs_indices = get_gs_indices(n_elec // 2, n_qubits - n_elec // 2)
    for i, (j, k) in enumerate(gs_indices):
        theta = params[i]
        unitary = get_gs_unitary(theta)
        try:
            name = f"Givens-SWAP({theta:.4f})"
        except TypeError:
            # jax tracer can't be formatted
            name = "Givens-SWAP"
        circuit.any(j, k, unitary=unitary, name=name)
    return circuit


@partial(jit, static_argnums=[1, 2, 3, 4, 5])
def get_statevector_tensornetwork(params, n_qubits, n_elec, ex_ops, param_ids, hcb=False, init_state=None):
    if tc.backend.name == "jax":
        logger.info(f"Entering `get_statevector_tensornetwork`. n_qubit: {n_qubits}")

    circuit = get_circuit(params, n_qubits, n_elec, ex_ops, param_ids, hcb, init_state)
    return circuit.state().real.reshape(-1)
