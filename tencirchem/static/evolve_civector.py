#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.


from functools import partial
import logging
from typing import Tuple

import numpy as np
from openfermion import jordan_wigner
import tensorcircuit as tc

from tencirchem.utils.backend import jit, fori_loop, scan, get_xp, get_uint_type
from tencirchem.utils.misc import ex_op_to_fop
from tencirchem.static.hamiltonian import apply_op
from tencirchem.static.ci_utils import get_ci_strings, get_addr, get_init_civector


logger = logging.getLogger(__name__)


def get_fket_permutation(f_idx, n_qubits, n_elec, ci_strings, strs2addr, hcb):
    mask = 0
    for i in f_idx:
        mask += 1 << i
    excitation = ci_strings ^ mask
    return get_addr(excitation, n_qubits, n_elec, strs2addr, hcb)


def get_fket_phase(f_idx, ci_strings):
    if len(f_idx) == 2:
        mask1 = 1 << f_idx[0]
        mask2 = 1 << f_idx[1]
    else:
        assert len(f_idx) == 4
        mask1 = (1 << f_idx[0]) + (1 << f_idx[1])
        mask2 = (1 << f_idx[2]) + (1 << f_idx[3])
    flip = ci_strings ^ mask1
    mask = mask1 | mask2
    masked = flip & mask
    positive = masked == mask
    negative = masked == 0
    return positive, negative


FERMION_PHASE_MASK_CACHE = {}


def get_fermion_phase(f_idx, n_qubits, ci_strings):
    if f_idx in FERMION_PHASE_MASK_CACHE:
        mask, sign = FERMION_PHASE_MASK_CACHE[f_idx]
    else:
        # fermion operator index, not sorted
        fop = ex_op_to_fop(f_idx)

        # pauli string index, already sorted
        qop = jordan_wigner(fop)
        mask_str = ["0"] * n_qubits
        for idx, term in next(iter(qop.terms.keys())):
            if term != "Z":
                assert idx in f_idx
                continue
            mask_str[n_qubits - 1 - idx] = "1"
        mask = get_uint_type()(int("".join(mask_str), base=2))

        if sorted(qop.terms.items())[0][1].real > 0:
            sign = -1
        else:
            sign = 1

        FERMION_PHASE_MASK_CACHE[f_idx] = mask, sign

    parity = ci_strings & mask
    assert parity.dtype in [np.uint32, np.uint64]
    if parity.dtype == np.uint32:
        mask = 0x11111111
        shift = 28
    else:
        mask = 0x1111111111111111
        shift = 60
    parity ^= parity >> 1
    parity ^= parity >> 2
    parity = (parity & mask) * mask
    parity = (parity >> shift) & 1

    return sign * np.sign(parity - 0.5).astype(np.int8)


def get_operators(n_qubits, n_elec, strs2addr, f_idx, ci_strings, hcb):
    if len(set(f_idx)) != len(f_idx):
        raise ValueError(f"Excitation {f_idx} not supported")
    xp = get_xp(tc.backend)
    fket_permutation = get_fket_permutation(f_idx, n_qubits, n_elec, ci_strings, strs2addr, hcb)
    fket_phase = xp.zeros(len(ci_strings))
    positive, negative = get_fket_phase(f_idx, ci_strings)
    fket_phase -= positive
    fket_phase += negative
    if not hcb:
        fket_phase *= get_fermion_phase(f_idx, n_qubits, ci_strings)
    f2ket_phase = xp.zeros(len(ci_strings))
    f2ket_phase -= positive
    f2ket_phase -= negative

    return fket_permutation, fket_phase, f2ket_phase


CI_OPERATOR_BATCH_CACHE = {}
CI_OPERATOR_CACHE = {}


@partial(jit, static_argnums=[0, 1, 2, 3])
def get_operator_tensors(n_qubits, n_elec, ex_ops, hcb=False):
    xp = get_xp(tc.backend)
    batch_key = (xp, tc.rdtypestr, n_qubits, n_elec, ex_ops, hcb)
    is_jax_backend = tc.backend.name == "jax"
    if not is_jax_backend and batch_key in CI_OPERATOR_BATCH_CACHE:
        return CI_OPERATOR_BATCH_CACHE[batch_key]

    ci_strings, strs2addr = get_ci_strings(n_qubits, n_elec, hcb, strs2addr=True)

    xp = get_xp(tc.backend)
    fket_permutation_tensor = xp.zeros((len(ex_ops), len(ci_strings)), dtype=get_uint_type())
    fket_phase_tensor = xp.zeros((len(ex_ops), len(ci_strings)), dtype=np.int8)
    f2ket_phase_tensor = xp.zeros((len(ex_ops), len(ci_strings)), dtype=np.int8)
    for i, f_idx in enumerate(ex_ops):
        if 64 < len(ex_ops):
            logger.info((i, f_idx))
        op_key = (xp, tc.rdtypestr, n_qubits, n_elec, f_idx, hcb)
        if not is_jax_backend and op_key in CI_OPERATOR_CACHE:
            fket_permutation, fket_phase, f2ket_phase = CI_OPERATOR_CACHE[op_key]
        else:
            fket_permutation, fket_phase, f2ket_phase = get_operators(
                n_qubits, n_elec, strs2addr, f_idx, ci_strings, hcb
            )
            if not is_jax_backend:
                CI_OPERATOR_CACHE[op_key] = fket_permutation, fket_phase, f2ket_phase
        fket_permutation_tensor[i] = fket_permutation
        fket_phase_tensor[i] = fket_phase
        f2ket_phase_tensor[i] = f2ket_phase

    fket_permutation_tensor = tc.backend.convert_to_tensor(fket_permutation_tensor)
    fket_phase_tensor = tc.backend.convert_to_tensor(fket_phase_tensor)
    f2ket_phase_tensor = tc.backend.convert_to_tensor(f2ket_phase_tensor)

    ret = ci_strings, fket_permutation_tensor, fket_phase_tensor, f2ket_phase_tensor
    if not is_jax_backend:
        CI_OPERATOR_BATCH_CACHE[batch_key] = ret
    return ret


@partial(jit, static_argnums=[1])
def get_theta_tensors(params, param_ids):
    theta_list = []
    for param_id in param_ids:
        theta_list.append(params[param_id])

    theta_tensor = tc.backend.convert_to_tensor(theta_list)
    theta_sin_tensor = tc.backend.sin(theta_tensor)
    theta_1mcos_tensor = 1 - tc.backend.cos(theta_tensor)
    return theta_sin_tensor, theta_1mcos_tensor


@jit
def evolve_civector_by_tensor(
    civector, fket_permutation_tensor, fket_phase_tensor, f2ket_phase_tensor, theta_sin, theta_1mcos
):
    def _evolve_excitation(j, _civector):
        _fket_phase = fket_phase_tensor[j]
        _fket_permutation = fket_permutation_tensor[j]
        fket = _civector[_fket_permutation] * _fket_phase
        f2ket = f2ket_phase_tensor[j] * _civector
        _civector += theta_1mcos[j] * f2ket + theta_sin[j] * fket
        return _civector

    return fori_loop(0, len(fket_permutation_tensor), _evolve_excitation, civector)


@partial(jit, static_argnums=[1, 2, 3, 4, 5])
def get_civector(params, n_qubits, n_elec, ex_ops, param_ids, hcb=False, init_state=None):
    if tc.backend.name == "jax":
        logger.info(f"Entering `get_civector`. n_qubit: {n_qubits}")

    ci_strings, fket_permutation_tensor, fket_phase_tensor, f2ket_phase_tensor = get_operator_tensors(
        n_qubits, n_elec, ex_ops, hcb
    )
    theta_sin, theta_1mcos = get_theta_tensors(params, param_ids)

    if init_state is None:
        civector = get_init_civector(len(ci_strings))
    else:
        civector = tc.backend.convert_to_tensor(init_state)
    civector = evolve_civector_by_tensor(
        civector, fket_permutation_tensor, fket_phase_tensor, f2ket_phase_tensor, theta_sin, theta_1mcos
    )

    return civector.reshape(-1)


def get_energy_and_grad_civector(
    params, hamiltonian, n_qubits, n_elec, ex_ops: Tuple, param_ids: Tuple, hcb: bool = False, init_state=None
):
    ket = get_civector(params, n_qubits, n_elec, ex_ops, param_ids, hcb, init_state)
    bra = apply_op(hamiltonian, ket)
    energy = bra @ ket
    # already cached
    op_tensors = get_operator_tensors(n_qubits, n_elec, ex_ops, hcb=hcb)
    theta_tensors = get_theta_tensors(params, param_ids)
    op_tensors = list(op_tensors) + list(theta_tensors)
    gradients_beforesum = _get_gradients_civector(bra, ket, *op_tensors[1:])

    gradients_beforesum = tc.backend.numpy(gradients_beforesum)
    gradients = np.zeros(params.shape)
    for grad, param_id in zip(gradients_beforesum, param_ids):
        gradients[param_id] += grad

    return energy, 2 * gradients


@jit
def _get_gradients_civector(
    bra, ket, fket_permutation_tensor, fket_phase_tensor, f2ket_phase_tensor, theta_sin, theta_1mcos
):
    scan_xs = fket_permutation_tensor, fket_phase_tensor, f2ket_phase_tensor, -theta_sin, theta_1mcos

    def _evolve_excitation(_civector, _fket_permutation, _fket_phase, _f2ket_phase):
        _civector += _f2ket_phase * _civector + _civector[_fket_permutation] * _fket_phase
        return _civector

    def get_grad(braket, scan_x):
        _bra, _ket = braket
        _fket_permutation, _fket_phase, _f2ket_phase, _theta_msin, _theta_1mcos = scan_x
        _ket = _evolve_excitation(_ket, _fket_permutation, _fket_phase * _theta_msin, _f2ket_phase * _theta_1mcos)
        _bra = _evolve_excitation(_bra, _fket_permutation, _fket_phase * _theta_msin, _f2ket_phase * _theta_1mcos)
        _fket = _ket[_fket_permutation] * _fket_phase
        grad = _bra @ _fket

        return (_bra, _ket), grad

    _, gradients = scan(get_grad, (bra, ket), scan_xs, len(fket_permutation_tensor), reverse=True)

    return gradients


def evolve_excitation_nocache(civector, fket_permutation, fket_phase, f2ket_phase, theta_1mcos, theta_sin):
    fket = civector[fket_permutation] * fket_phase
    f2ket = civector * f2ket_phase
    civector += theta_1mcos * f2ket + theta_sin * fket
    return civector


@partial(jit, static_argnums=[1, 2, 3, 4, 5])
def get_civector_nocache(params, n_qubits, n_elec, ex_ops, param_ids, hcb=False, init_state=None):
    if tc.backend.name == "jax":
        logger.info(f"Entering `get_civector_nocache`. n_qubit: {n_qubits}")

    theta_sin_tensor, theta_1mcos_tensor = get_theta_tensors(params, param_ids)
    ci_strings, strs2addr = get_ci_strings(n_qubits, n_elec, hcb, strs2addr=True)

    if init_state is None:
        civector = get_init_civector(len(ci_strings))
    else:
        civector = tc.backend.convert_to_tensor(init_state)

    for theta_sin, theta_1mcos, f_idx in zip(theta_sin_tensor, theta_1mcos_tensor, ex_ops):
        fket_permutation, fket_phase, f2ket_phase = get_operators(n_qubits, n_elec, strs2addr, f_idx, ci_strings, hcb)
        civector = evolve_excitation_nocache(
            civector, fket_permutation, fket_phase, f2ket_phase, theta_1mcos, theta_sin
        )

    return civector.reshape(-1)


def get_energy_and_grad_civector_nocache(
    params, hamiltonian, n_qubits, n_elec, ex_ops: Tuple, param_ids: Tuple, hcb: bool = False, init_state=None
):
    ket = get_civector_nocache(params, n_qubits, n_elec, ex_ops, param_ids, hcb, init_state)
    bra = apply_op(hamiltonian, ket)
    energy = bra @ ket

    gradients_beforesum = _get_gradients_civector_nocache(bra, ket, params, n_qubits, n_elec, ex_ops, param_ids, hcb)
    gradients_beforesum = tc.backend.numpy(gradients_beforesum)

    gradients = np.zeros(params.shape)
    for grad, param_id in zip(gradients_beforesum, param_ids):
        gradients[param_id] += grad

    return energy, 2 * gradients


@partial(jit, static_argnums=[3, 4, 5, 6, 7])
def _get_gradients_civector_nocache(bra, ket, params, n_qubits, n_elec, ex_ops, param_ids, hcb):
    ci_strings, strs2addr = get_ci_strings(n_qubits, n_elec, hcb, True)
    theta_sin_tensor, theta_1mcos_tensor = get_theta_tensors(params, param_ids)

    gradients_beforesum = []
    for theta_sin, theta_1mcos, f_idx in reversed(list(zip(theta_sin_tensor, theta_1mcos_tensor, ex_ops))):
        fket_permutation, fket_phase, f2ket_phase = get_operators(n_qubits, n_elec, strs2addr, f_idx, ci_strings, hcb)
        bra = evolve_excitation_nocache(bra, fket_permutation, fket_phase, f2ket_phase, theta_1mcos, -theta_sin)
        ket = evolve_excitation_nocache(ket, fket_permutation, fket_phase, f2ket_phase, theta_1mcos, -theta_sin)
        fket = ket[fket_permutation] * fket_phase
        grad = bra @ fket
        gradients_beforesum.append(grad)
    gradients_beforesum = list(reversed(gradients_beforesum))
    gradients_beforesum = tc.backend.convert_to_tensor(gradients_beforesum)

    return gradients_beforesum


def apply_excitation_civector(civector, n_qubits, n_elec, f_idx, hcb):
    _, fket_permutation, fket_phase, _ = get_operator_tensors(n_qubits, n_elec, ex_ops=(f_idx,), hcb=hcb)
    return civector[fket_permutation[0]] * fket_phase[0]


def apply_excitation_civector_nocache(civector, n_qubits, n_elec, f_idx, hcb):
    ci_strings, strs2addr = get_ci_strings(n_qubits, n_elec, hcb, strs2addr=True)
    fket_permutation, fket_phase, _ = get_operators(n_qubits, n_elec, strs2addr, f_idx, ci_strings, hcb)
    return civector[fket_permutation] * fket_phase
