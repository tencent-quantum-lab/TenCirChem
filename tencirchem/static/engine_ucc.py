#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.


from functools import partial
from typing import Tuple
import logging

import tensorcircuit as tc

from tencirchem.utils.backend import jit, value_and_grad
from tencirchem.static.hamiltonian import apply_op
from tencirchem.static.ci_utils import get_ci_strings, civector_to_statevector, statevector_to_civector
from tencirchem.static.evolve_civector import (
    get_civector_nocache,
    get_energy_and_grad_civector,
    get_energy_and_grad_civector_nocache,
    apply_excitation_civector,
    apply_excitation_civector_nocache,
)
from tencirchem.static.evolve_civector import get_civector as get_civector_
from tencirchem.static.evolve_statevector import apply_excitation_statevector
from tencirchem.static.evolve_statevector import get_statevector as get_statevector_
from tencirchem.static.evolve_tensornetwork import get_statevector_tensornetwork
from tencirchem.static.evolve_pyscf import get_civector_pyscf, get_energy_and_grad_pyscf, apply_excitation_pyscf


logger = logging.getLogger(__name__)


GETVECTOR_MAP = {
    "tensornetwork": get_statevector_tensornetwork,
    "statevector": get_statevector_,
    "civector": get_civector_,
    "civector-large": get_civector_nocache,
    "pyscf": get_civector_pyscf,
}


def get_ket(params, n_qubits, n_elec, ex_ops, param_ids, hcb, init_state, ci_strings, engine):
    if param_ids is None:
        param_ids = range(len(ex_ops))
    func = GETVECTOR_MAP[engine]
    init_state = translate_init_state(init_state, n_qubits, ci_strings)
    ket = func(
        params,
        n_qubits=n_qubits,
        n_elec=n_elec,
        ex_ops=tuple(ex_ops),
        param_ids=tuple(param_ids),
        hcb=hcb,
        init_state=init_state,
    )
    return ket


def get_civector(params, n_qubits, n_elec, ex_ops, param_ids, hcb, init_state, engine):
    ci_strings = get_ci_strings(n_qubits, n_elec, hcb)
    ket = get_ket(params, n_qubits, n_elec, ex_ops, param_ids, hcb, init_state, ci_strings, engine)
    if engine.startswith("civector") or engine == "pyscf":
        civector = ket
    else:
        civector = ket[ci_strings]
    return civector


def get_statevector(params, n_qubits, n_elec, ex_ops, param_ids, hcb, init_state, engine):
    ci_strings = get_ci_strings(n_qubits, n_elec, hcb)
    ket = get_ket(params, n_qubits, n_elec, ex_ops, param_ids, hcb, init_state, ci_strings, engine)
    if engine.startswith("civector") or engine == "pyscf":
        statevector = civector_to_statevector(ket, n_qubits, ci_strings)
    else:
        statevector = ket
    return statevector


def get_energy(params, hamiltonian, n_qubits, n_elec, ex_ops: Tuple, param_ids: Tuple, hcb: bool, init_state, engine):
    if param_ids is None:
        param_ids = range(len(ex_ops))
    logger.info(f"Entering `get_energy`")
    ci_strings = get_ci_strings(n_qubits, n_elec, hcb)
    init_state = translate_init_state(init_state, n_qubits, ci_strings)
    ket = GETVECTOR_MAP[engine](
        params, n_qubits, n_elec, tuple(ex_ops), tuple(param_ids), hcb=hcb, init_state=init_state
    )
    hket = apply_op(hamiltonian, ket)
    return ket @ hket


get_energy_statevector = partial(get_energy, engine="statevector")
try:
    get_energy_and_grad_statevector = jit(value_and_grad(get_energy_statevector), static_argnums=[2, 3, 4, 5, 6])
except NotImplementedError:

    def get_energy_and_grad_statevector(*args, **kwargs):
        raise NotImplementedError("Non JAX-backend for statevector engine")


get_energy_tensornetwork = partial(get_energy, engine="tensornetwork")
try:
    get_energy_and_grad_tensornetwork = jit(value_and_grad(get_energy_tensornetwork), static_argnums=[2, 3, 4, 5, 6])
except NotImplementedError:

    def get_energy_and_grad_tensornetwork(*args, **kwargs):
        raise NotImplementedError("Non JAX-backend for tensornetwork engine")


ENERGY_AND_GRAD_MAP = {
    "tensornetwork": get_energy_and_grad_tensornetwork,
    "statevector": get_energy_and_grad_statevector,
    "civector": get_energy_and_grad_civector,
    "civector-large": get_energy_and_grad_civector_nocache,
    "pyscf": get_energy_and_grad_pyscf,
}


def get_energy_and_grad(params, hamiltonian, n_qubits, n_elec, ex_ops, param_ids, hcb, init_state, engine):
    if engine not in ENERGY_AND_GRAD_MAP:
        raise ValueError(f"Engine '{engine}' not supported")

    func = ENERGY_AND_GRAD_MAP[engine]
    ci_strings = get_ci_strings(n_qubits, n_elec, hcb)
    init_state = translate_init_state(init_state, n_qubits, ci_strings)
    return func(params, hamiltonian, n_qubits, n_elec, tuple(ex_ops), tuple(param_ids), hcb, init_state)


APPLY_EXCITATION_MAP = {
    # share the same function with statevector engine
    "tensornetwork": apply_excitation_statevector,
    "statevector": apply_excitation_statevector,
    "civector": apply_excitation_civector,
    "civector-large": apply_excitation_civector_nocache,
    "pyscf": apply_excitation_pyscf,
}


def apply_excitation(state, n_qubits, n_elec, ex_op, hcb, engine):
    if engine not in APPLY_EXCITATION_MAP:
        raise ValueError(f"Engine '{engine}' not supported")

    state = tc.backend.convert_to_tensor(state)

    is_statevector_input = len(state) == (1 << n_qubits)
    is_statevector_engine = engine in ["tensornetwork", "statevector"]

    if is_statevector_input and not is_statevector_engine:
        ci_strings = get_ci_strings(n_qubits, n_elec, hcb)
        state = statevector_to_civector(state, ci_strings)
    if not is_statevector_input and is_statevector_engine:
        ci_strings = get_ci_strings(n_qubits, n_elec, hcb)
        state = civector_to_statevector(state, n_qubits, ci_strings)
    func = APPLY_EXCITATION_MAP[engine]
    res_state = func(state, n_qubits, n_elec, ex_op, hcb)
    if is_statevector_input and not is_statevector_engine:
        return civector_to_statevector(res_state, n_qubits, ci_strings)
    if not is_statevector_input and is_statevector_engine:
        return statevector_to_civector(res_state, ci_strings)
    return res_state


def translate_init_state(init_state, n_qubits, ci_strings):
    if init_state is None:
        return None
    # translate to civector first for all engines to be JAX-compatible
    if isinstance(init_state, tc.Circuit):
        # note no cupy backend for tc
        init_state = statevector_to_civector(tc.backend.convert_to_tensor(init_state.state().real), ci_strings)
    else:
        is_statevector_input = len(init_state) == (1 << n_qubits)
        if is_statevector_input:
            init_state = statevector_to_civector(init_state, ci_strings)
    return tc.backend.convert_to_tensor(init_state)
