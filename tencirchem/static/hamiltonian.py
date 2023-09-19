#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.


import logging
from inspect import isfunction
from itertools import product
from typing import Tuple

import numpy as np
import tensornetwork as tn
from renormalizer import Mpo
from openfermion import FermionOperator, QubitOperator
from pyscf.scf.hf import RHF
from pyscf.mcscf import CASCI
from pyscf.fci import direct_nosym, cistring
from pyscf import ao2mo
import tensorcircuit as tc
from tensorcircuit import QuOperator

from tencirchem.utils.misc import hcb_to_coo, fop_to_coo, reverse_qop_idx, canonical_mo_coeff, get_n_qubits
from tencirchem.constants import DISCARD_EPS


logger = logging.getLogger(__name__)


def get_integral_from_hf(hf: RHF, active_space: Tuple = None):
    if not isinstance(hf, RHF):
        raise TypeError(f"hf object must be RHF class, got {type(hf)}")
    m = hf.mol
    assert hf.mo_coeff is not None
    # todo(weitangli): don't modify inplace
    hf.mo_coeff = canonical_mo_coeff(hf.mo_coeff)

    if active_space is None:
        nelecas = m.nelectron
        ncas = m.nao
    else:
        nelecas, ncas = active_space

    casci = CASCI(hf, ncas, nelecas)
    int1e, e_core = casci.get_h1eff()
    int2e = ao2mo.restore("s1", casci.get_h2eff(), ncas)

    return int1e, int2e, e_core


def get_hop_from_integral(int1e, int2e):
    n_orb = int1e.shape[0]
    if int1e.shape != (n_orb, n_orb):
        raise ValueError(f"Invalid one-boby integral array shape: {int1e.shape}")
    int2e = ao2mo.restore(1, int2e, n_orb)
    assert int2e.shape == (n_orb, n_orb, n_orb, n_orb)
    n_sorb = n_orb * 2

    logger.info("Creating Hamiltonian operators")

    h1e = np.zeros((n_sorb, n_sorb))
    h2e = np.zeros((n_sorb, n_sorb, n_sorb, n_sorb))

    h1e[:n_orb, :n_orb] = h1e[n_orb:, n_orb:] = int1e

    for p, q, r, s in product(range(n_sorb), repeat=4):
        # a_p^\dagger a_q^\dagger a_r a_s
        if ((p < n_orb) == (s < n_orb)) and ((q < n_orb) == (r < n_orb)):
            # note the different orders of the indices
            h2e[p, q, r, s] = int2e[p % n_orb, s % n_orb, q % n_orb, r % n_orb]

    op1e = []
    for p, q in product(range(n_sorb), repeat=2):
        # a_p^\dagger a_q
        v = h1e[p, q]
        if np.abs(v) < DISCARD_EPS:
            continue
        op = FermionOperator(f"{p}^ {q}", v)
        op1e.append(op)

    op2e = []
    for q, s in product(range(n_sorb), repeat=2):
        for p, r in product(range(q), range(s)):
            # a_p^\dagger a_q^\dagger a_r a_s
            v = h2e[p, q, r, s] - h2e[q, p, r, s]
            if np.abs(v) < DISCARD_EPS:
                continue
            op = FermionOperator(f"{p}^ {q}^ {r} {s}", v)
            op2e.append(op)

    logger.info("Summing Hamiltonian operators")
    ops = FermionOperator()
    for op in op1e + op2e:
        ops += op

    return ops


def qubit_operator(string: str, coeff: float) -> QubitOperator:
    ret = coeff
    terms = string.split(" ")
    for term in terms:
        if term[-1] == "^":
            sign = -1
            term = term[:-1]
        else:
            sign = 1
        idx = int(term)
        ret *= (QubitOperator(f"X{idx}") + sign * 1j * QubitOperator(f"Y{idx}")) / 2
    return ret


def get_hop_hcb_from_integral(int1e, int2e):
    # Hard core boson Hamiltonian
    # https://arxiv.org/pdf/2002.00035.pdf
    n_orb = int1e.shape[0]
    qop = QubitOperator()
    for p in range(n_orb):
        for q in range(p + 1):
            if p == q:
                qop += qubit_operator(f"{p}^ {p}", 2 * int1e[p, p] + int2e[p, p, p, p])
            else:
                qop += qubit_operator(f"{p}^ {q}", int2e[p, q, p, q])
                qop += qubit_operator(f"{q}^ {p}", int2e[q, p, q, p])
                qop += qubit_operator(f"{p}^ {p} {q}^ {q}", 4 * int2e[p, p, q, q] - 2 * int2e[p, q, p, q])
    qop = reverse_qop_idx(qop, n_orb)
    return qop


def get_h_sparse_from_integral(int1e, int2e, hcb=False, do_log=False):
    if not hcb:
        ops = get_hop_from_integral(int1e, int2e)
    else:
        ops = get_hop_hcb_from_integral(int1e, int2e)
    if do_log:
        logger.info("Constructing sparse Hamiltonian")
    if not hcb:
        h_sparse = fop_to_coo(ops, n_qubits=2 * len(int1e))
    else:
        h_sparse = hcb_to_coo(ops, n_qubits=len(int1e))
    if do_log:
        logger.info("Sparse Hamiltonian constructed")
    return h_sparse


def get_h_fcifunc_from_integral(int1e, int2e, n_elec):
    n_orb = len(int1e)
    h2e = direct_nosym.absorb_h1e(int1e, int2e, n_orb, n_elec, 0.5)

    def fci_func(civector):
        civector = tc.backend.numpy(civector).astype(np.float64)
        civector = direct_nosym.contract_2e(h2e, civector, norb=n_orb, nelec=n_elec)
        return tc.backend.convert_to_tensor(civector).astype(tc.rdtypestr)

    return fci_func


def get_h_fcifunc_hcb_from_integral(int1e, int2e, n_elec):
    # todo: how about using https://github.com/pyscf/doci
    n_orb = len(int1e)
    ci_strings = cistring.make_strings(range(n_orb), n_elec // 2)

    def fci_func(civector):
        res = tc.backend.zeros(len(civector), dtype=tc.rdtypestr)
        for p in range(n_orb):
            for q in range(p + 1):
                if p == q:
                    bitmask = 1 << p
                    arraymask = (ci_strings & bitmask) == bitmask
                    res += (civector * arraymask) * (2 * int1e[p, p] + int2e[p, p, p, p])
                else:
                    bitmask = (1 << p) + (1 << q)
                    excitation = ci_strings ^ bitmask
                    addr = cistring.strs2addr(n_orb, n_elec // 2, excitation)
                    flip = ci_strings ^ (1 << p)
                    masked_flip = flip & bitmask
                    arraymask = (masked_flip == bitmask) | (masked_flip == 0)
                    res += civector[addr] * arraymask * int2e[p, q, p, q]
                    arraymask = (ci_strings & bitmask) == bitmask
                    res += (civector * arraymask) * (4 * int2e[p, p, q, q] - 2 * int2e[p, q, p, q])
        return res

    return fci_func


def get_h_from_integral(int1e, int2e, n_elec, hcb: bool, htype: str):
    if htype == "sparse":
        hamiltonian = get_h_sparse_from_integral(int1e, int2e, hcb=hcb, do_log=True)
    else:
        assert htype.lower() == "fcifunc"
        if not hcb:
            hamiltonian = get_h_fcifunc_from_integral(int1e, int2e, n_elec)
        else:
            hamiltonian = get_h_fcifunc_hcb_from_integral(int1e, int2e, n_elec)
    return hamiltonian


def get_h_from_hf(hf: RHF, active_space: Tuple = None, hcb: bool = False, htype="sparse"):
    if not isinstance(hf, RHF):
        raise TypeError(f"hf object must RHF class, got {type(hf)}")
    htype = htype.lower()
    if not htype in ["sparse", "mpo", "fcifunc"]:
        raise ValueError(f"htype must be 'sparse' or 'mpo', got '{htype}'")
    int1e, int2e, e_core = get_integral_from_hf(hf, active_space)
    if active_space is None:
        n_elec = hf.mol.nelectron
    else:
        n_elec = active_space[0]

    hamiltonian = get_h_from_integral(int1e, int2e, n_elec, hcb, htype)

    if active_space is None:
        return hamiltonian
    else:
        return hamiltonian, e_core


def mpo_to_quoperator(mpo: Mpo):
    array_list = [m.array for m in mpo]
    # squeeze dim 1 out
    assert len(array_list) >= 2
    a, b, c, d = array_list[0].shape
    array_list[0] = array_list[0].reshape(b, c, d)
    a, b, c, d = array_list[-1].shape
    array_list[-1] = array_list[-1].reshape(a, b, c)
    # convert MPO to tensor-network
    node_list = [tn.Node(array) for array in array_list]

    node_list[0][2] ^ node_list[1][0]
    for i in range(1, len(node_list) - 1):
        node_list[i][3] ^ node_list[i + 1][0]

    in_edges = [node_list[0][0]] + [node[1] for node in node_list[1:]]
    out_edges = [node_list[0][1]] + [node[2] for node in node_list[1:]]

    qop = QuOperator(out_edges, in_edges)
    return qop


def apply_op(op, state):
    if isinstance(op, list):
        # in MPO form
        n_qubit = get_n_qubits(state)
        n_qubit_op = get_n_qubits(op)
        assert n_qubit == n_qubit_op
        mps = tc.QuVector.from_tensor(state.reshape([2] * n_qubit))
        h_qop = mpo_to_quoperator(op)
        return (h_qop @ mps).eval().reshape(-1)
    if isfunction(op):
        return op(state)
    else:
        return op @ state


def random_integral(nao: int, seed: int = 2077):
    np.random.seed(seed)
    int1e = np.random.uniform(-1, 1, size=(nao, nao))
    int2e = np.random.uniform(-1, 1, size=(nao, nao, nao, nao))
    int1e = 0.5 * (int1e + int1e.T)
    int2e = symmetrize_int2e(int2e)
    return int1e, int2e


def symmetrize_int2e(int2e):
    int2e = 0.25 * (
        int2e + int2e.transpose((0, 1, 3, 2)) + int2e.transpose((1, 0, 2, 3)) + int2e.transpose((2, 3, 0, 1))
    )
    int2e = 0.5 * (int2e + int2e.transpose(3, 2, 1, 0))
    return int2e
