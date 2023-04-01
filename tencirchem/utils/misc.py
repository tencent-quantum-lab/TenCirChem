#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.


from functools import wraps
from inspect import isfunction
from typing import List, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from renormalizer import Model, Mpo, Op
from renormalizer.model.basis import BasisSet
from openfermion import FermionOperator, QubitOperator, jordan_wigner, get_sparse_operator
from openfermion.utils import hermitian_conjugated
from qiskit.quantum_info import SparsePauliOp
import tensorcircuit as tc

from tencirchem.constants import DISCARD_EPS


def csc_to_coo(csc):
    coo = coo_matrix(csc)
    mask = DISCARD_EPS < np.abs(coo.data.real)
    indices = np.array([coo.row[mask], coo.col[mask]]).T
    values = coo.data.real[mask].astype(tc.rdtypestr)
    return tc.backend.coo_sparse_matrix(indices=indices, values=values, shape=coo.shape)


def fop_to_coo(fop: FermionOperator, n_qubits: int, real: bool = True):
    op = get_sparse_operator(jordan_wigner(reverse_fop_idx(fop, n_qubits)), n_qubits=n_qubits)
    if real:
        op = op.real
    return csc_to_coo(op)


def hcb_to_coo(qop: QubitOperator, n_qubits: int, real: bool = True):
    op = get_sparse_operator(qop, n_qubits)
    if real:
        op = op.real
    return csc_to_coo(op)


def qop_to_qiskit(qop: QubitOperator, n_qubits: int) -> SparsePauliOp:
    sparse_list = []
    for k, v in qop.terms.items():
        s = "".join(kk[1] for kk in k)
        idx = [kk[0] for kk in k]
        sparse_list.append([s, idx, v])
    return SparsePauliOp.from_sparse_list(sparse_list, num_qubits=n_qubits)


def reverse_qop_idx(op: QubitOperator, n_qubits: int) -> QubitOperator:
    ret = QubitOperator()
    for pauli_string, v in op.terms.items():
        # internally QubitOperator assumes ascending index
        pauli_string = tuple(reversed([(n_qubits - 1 - idx, symbol) for idx, symbol in pauli_string]))
        ret.terms[pauli_string] = v
    return ret


def reverse_fop_idx(op: FermionOperator, n_qubits: int) -> FermionOperator:
    ret = FermionOperator()
    for word, v in op.terms.items():
        word = tuple([(n_qubits - 1 - idx, symbol) for idx, symbol in word])
        ret.terms[word] = v
    return ret


def format_ex_op(ex_op: Tuple) -> str:
    if len(ex_op) == 2:
        return f"{ex_op[0]}^ {ex_op[1]}"
    else:
        assert len(ex_op) == 4
        return f"{ex_op[0]}^ {ex_op[1]}^ {ex_op[2]} {ex_op[3]}"


def scipy_opt_wrap(f, gradient=True):
    @wraps(f)
    def _wrap_scipy_opt(_params, *args):
        # scipy assumes 64bit https://github.com/scipy/scipy/issues/5832
        res = f(tc.backend.convert_to_tensor(_params), *args)
        if gradient:
            return [np.asarray(tc.backend.numpy(v), dtype=np.float64) for v in res]
        else:
            return np.asarray(tc.backend.numpy(res), dtype=np.float64)

    return _wrap_scipy_opt


def rdm_mo2ao(rdm: np.ndarray, mo_coeff: np.ndarray):
    c = mo_coeff
    if rdm.ndim == 2:
        return c @ rdm @ c.T
    else:
        assert rdm.ndim == 4
        for _ in range(4):
            rdm = np.tensordot(rdm, c.T, axes=1).transpose(3, 0, 1, 2)
        return rdm


def canonical_mo_coeff(mo_coeff: np.ndarray):
    # make the first large element positive
    # all elements smaller than 1e-5 is highly unlikely (at least 1e10 basis)
    largest_elem_idx = np.argmax(1e-5 < np.abs(mo_coeff), axis=0)
    largest_elem = mo_coeff[(largest_elem_idx, np.arange(len(largest_elem_idx)))]
    return mo_coeff * np.sign(largest_elem).reshape(1, -1)


def get_n_qubits(vector_or_matrix_or_mpo_func):
    if isinstance(vector_or_matrix_or_mpo_func, list):
        return len(vector_or_matrix_or_mpo_func)
    if isfunction(vector_or_matrix_or_mpo_func):
        return vector_or_matrix_or_mpo_func.n_qubit
    # if hasattr(vector_or_matrix_or_mpo_func, "n_qubits"):
    #     return getattr(vector_or_matrix_or_mpo_func, "n_qubits")
    return round(np.log2(vector_or_matrix_or_mpo_func.shape[0]))


def ex_op_to_fop(ex_op, with_conjugation=False):
    if len(ex_op) == 2:
        fop = FermionOperator(f"{ex_op[0]}^ {ex_op[1]}")
    else:
        assert len(ex_op) == 4
        fop = FermionOperator(f"{ex_op[0]}^ {ex_op[1]}^ {ex_op[2]} {ex_op[3]}")
    if with_conjugation:
        fop = fop - hermitian_conjugated(fop)
    return fop


def get_dense_operator(basis: List[BasisSet], terms: List[Op]):
    return Mpo(Model(basis, []), terms).todense()
