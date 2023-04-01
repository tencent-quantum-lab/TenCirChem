#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.


import logging
from functools import partial

import numpy as np
from openfermion import jordan_wigner
import tensorcircuit as tc

from tencirchem.utils.backend import jit
from tencirchem.constants import ad_a_hc2, adad_aa_hc2, ad_a_hc, adad_aa_hc
from tencirchem.utils.misc import ex_op_to_fop
from tencirchem.static.evolve_tensornetwork import get_init_circuit


logger = logging.getLogger(__name__)


@partial(jit, static_argnums=[1, 2, 3, 4, 5])
def get_statevector(params, n_qubits, n_elec, ex_ops, param_ids, hcb=False, init_state=None):
    if tc.backend.name == "jax":
        logger.info(f"Entering `get_statevector`. n_qubit: {n_qubits}")
    if param_ids is None:
        assert len(params) == len(ex_ops)
        param_ids = list(range(len(params)))

    circuit = get_init_circuit(n_qubits, n_elec, hcb, init_state)

    # note only the real part is taken
    statevector = tc.backend.convert_to_tensor(circuit.state())

    for param_id, f_idx in zip(param_ids, ex_ops):
        theta = params[param_id]
        statevector = evolve_excitation(statevector, f_idx, theta, hcb)

    return statevector.real.reshape(-1)


@partial(jit, static_argnums=[1, 3])
def evolve_excitation(statevector, f_idx, theta, hcb):
    n_qubits = round(np.log2(statevector.shape[0]))
    qubit_idx = [n_qubits - 1 - idx for idx in f_idx]
    # fermion operator operated on ket, twice
    f2ket = tc.Circuit(n_qubits, inputs=statevector)
    # fermion operator index, not sorted
    if len(qubit_idx) == 2:
        f2ket.any(*qubit_idx, unitary=ad_a_hc2)
    else:
        assert len(qubit_idx) == 4
        f2ket.any(*qubit_idx, unitary=adad_aa_hc2)

    # fermion operator operated on ket
    fket = apply_excitation(statevector, f_idx, hcb)
    statevector += (1 - tc.backend.cos(theta)) * f2ket.state() + tc.backend.sin(theta) * fket
    return statevector


@partial(jit, static_argnums=[1, 2])
def apply_excitation(statevector, f_idx, hcb):
    n_qubits = round(np.log2(statevector.shape[0]))
    qubit_idx = [n_qubits - 1 - idx for idx in f_idx]
    circuit = tc.Circuit(n_qubits, inputs=statevector)
    # fermion operator index, not sorted
    if len(qubit_idx) == 2:
        circuit.any(*qubit_idx, unitary=ad_a_hc)
    else:
        assert len(qubit_idx) == 4
        circuit.any(*qubit_idx, unitary=adad_aa_hc)

    if hcb:
        return circuit.state()

    # pauli string index, already sorted
    fop = ex_op_to_fop(f_idx)
    qop = jordan_wigner(fop)
    z_indices = []
    for idx, term in next(iter(qop.terms.keys())):
        if term != "Z":
            assert idx in f_idx
            continue
        z_indices.append(n_qubits - 1 - idx)
    if sorted(qop.terms.items())[0][1].real > 0:
        sign = 1
    else:
        sign = -1
    phase_vector = [sign]
    for i in range(n_qubits):
        if i in z_indices:
            phase_vector = np.kron(phase_vector, [1, -1])
        else:
            phase_vector = np.kron(phase_vector, [1, 1])
    return phase_vector * circuit.state()


# external interface
def apply_excitation_statevector(statevector, n_qubits, n_elec, f_idx, hcb):
    return apply_excitation(statevector, f_idx, hcb).real
