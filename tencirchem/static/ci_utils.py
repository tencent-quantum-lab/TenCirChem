#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.


from functools import partial

import numpy as np
from pyscf.fci import cistring
import tensorcircuit as tc

from tencirchem.utils.backend import jit, tensor_set_elem, get_xp, get_uint_type


def get_ci_strings(n_qubits, n_elec, hcb, strs2addr=False):
    xp = get_xp(tc.backend)
    uint_type = get_uint_type()
    if 2**n_qubits > np.iinfo(uint_type).max:
        raise ValueError(f"Too many qubits: {n_qubits}, try using complex128 datatype")
    if not hcb:
        beta = cistring.make_strings(range(n_qubits // 2), n_elec // 2)
        beta = xp.array(beta, dtype=uint_type)
        alpha = beta << (n_qubits // 2)
        ci_strings = (alpha.reshape(-1, 1) + beta.reshape(1, -1)).ravel()
        if strs2addr:
            strs2addr = xp.zeros(2 ** (n_qubits // 2), dtype=uint_type)
            strs2addr[beta] = xp.arange(len(beta))
            return ci_strings, strs2addr
    else:
        ci_strings = cistring.make_strings(range(n_qubits), n_elec // 2).astype(uint_type)
        if strs2addr:
            strs2addr = xp.zeros(2**n_qubits, dtype=uint_type)
            strs2addr[ci_strings] = xp.arange(len(ci_strings))
            return ci_strings, strs2addr

    return ci_strings


def get_addr(excitation, n_qubits, n_elec, strs2addr, hcb, num_strings=None):
    if hcb:
        return strs2addr[excitation]
    alpha = excitation >> (n_qubits // 2)
    beta = excitation & (2 ** (n_qubits // 2) - 1)
    alpha_addr2 = strs2addr[alpha]
    beta_addr2 = strs2addr[beta]
    if num_strings is None:
        num_strings = cistring.num_strings(n_qubits // 2, n_elec // 2)
    return alpha_addr2 * num_strings + beta_addr2


def get_ex_bitstring(n_qubits, n_elec, ex_op, hcb):
    if not hcb:
        bitstring_base = ["0"] * (n_qubits // 2 - n_elec // 2) + ["1"] * (n_elec // 2)
        bitstring_base *= 2
    else:
        bitstring_base = ["0"] * (n_qubits - n_elec // 2) + ["1"] * (n_elec // 2)

    bitstring = bitstring_base.copy()[::-1]
    # first annihilation then creation
    if len(ex_op) == 2:
        bitstring[ex_op[1]] = "0"
        bitstring[ex_op[0]] = "1"
    else:
        assert len(ex_op) == 4
        bitstring[ex_op[3]] = "0"
        bitstring[ex_op[2]] = "0"
        bitstring[ex_op[1]] = "1"
        bitstring[ex_op[0]] = "1"

    return "".join(reversed(bitstring))


def civector_to_statevector(civector, n_qubits, ci_strings):
    statevector = tc.backend.zeros(2**n_qubits, dtype=tc.rdtypestr)
    return tensor_set_elem(statevector, ci_strings, civector)


def statevector_to_civector(statevector, ci_strings):
    return statevector[ci_strings]


@partial(jit, static_argnums=[0])
def get_init_civector(len_ci):
    civector = tc.backend.zeros(len_ci, dtype=tc.rdtypestr)
    civector = tensor_set_elem(civector, 0, 1)
    return civector
