#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.


from typing import Any, List, Union, Tuple
import numpy as np

from renormalizer.model import Op, OpSum, Model
from renormalizer import BasisHalfSpin, BasisSimpleElectron, BasisMultiElectron, BasisMultiElectronVac, Mps
from renormalizer.model.basis import BasisSet
import tensorcircuit as tc

from tencirchem.constants import DISCARD_EPS


# to be added to new DOFs
DOF_SALT = "TCCQUBIT"


def check_basis_type(basis: List[BasisSet]):
    for b in basis:
        if isinstance(b, (BasisMultiElectronVac,)):
            raise TypeError(f"Unsupported basis: {type(b)}")
        if isinstance(b, BasisMultiElectron) and len(b.dofs) != 2:
            raise ValueError(f"For only two DOFs are allowed in BasisMultiElectron. Got {b}")


def qubit_encode_op(
    terms: Union[List[Op], Op], basis: List[BasisSet], boson_encoding: str = None
) -> Tuple[OpSum, float]:
    check_basis_type(basis)
    if isinstance(terms, Op):
        terms = [terms]

    model = Model(basis, [])

    new_terms = []
    for op in terms:
        terms, factor = op.split_elementary(model.dof_to_siteidx)
        opsum_list = []
        for op in terms:
            opsum = transform_op(op, model.dof_to_basis[op.dofs[0]], boson_encoding)
            opsum_list.append(opsum)

        new_term = 1
        for opsum in opsum_list:
            new_term = new_term * opsum
        new_term = new_term * factor

        new_terms.extend(new_term)

    # post process
    # pick out constants
    identity_terms: List[Op] = []
    non_identity_terms = OpSum()
    for op in new_terms:
        if op.is_identity:
            identity_terms.append(op)
        else:
            non_identity_terms.append(op.squeeze_identity())

    constant = sum([op.factor for op in identity_terms])

    return non_identity_terms.simplify(atol=DISCARD_EPS), constant


def qubit_encode_op_grouped(
    terms: List[Union[List[Op], Op]], basis: List[BasisSet], boson_encoding: str = None
) -> Tuple[List[OpSum], float]:
    new_terms = []
    constant_sum = 0
    for op in terms:
        opsum, constant = qubit_encode_op(op, basis, boson_encoding)
        new_terms.append(opsum)
        constant_sum += constant

    return new_terms, constant_sum


def qubit_encode_basis(basis: List[BasisSet], boson_encoding=None):
    spin_basis = []
    for b in basis:
        if isinstance(b, BasisMultiElectron):
            assert b.nbas == 2
            spin_basis.append(BasisHalfSpin(b.dofs))
        elif b.is_phonon:
            if boson_encoding is None:
                new_dofs = [b.dof]
            elif boson_encoding == "unary":
                new_dofs = [(b.dof, f"{DOF_SALT}-{i}") for i in range(b.nbas)]
            else:
                assert boson_encoding.lower() in ["binary", "gray"]
                n_qubits = int(np.ceil(np.log2(b.nbas)))
                new_dofs = [(b.dof, f"{DOF_SALT}-{i}") for i in range(n_qubits)]
            new_basis = [BasisHalfSpin(dof) for dof in new_dofs]
            spin_basis.extend(new_basis)
        else:
            spin_basis.append(BasisHalfSpin(b.dof))

    return spin_basis


def transform_op(op: Op, basis: BasisSet, boson_encoding: str = None) -> OpSum:
    # `op` is "elementary": all terms are in the `basis`
    assert op.factor == 1

    if set(op.split_symbol) == {"I"}:
        return OpSum([op])

    if isinstance(basis, (BasisHalfSpin, BasisSimpleElectron, BasisMultiElectron)):
        if isinstance(basis, BasisMultiElectron):
            assert len(basis.dof) == 2
            new_dof = basis.dofs
        else:
            new_dof = op.dofs[0]
        return transform_op_direct(op, new_dof, basis)

    # in principle can encode basis sets such as BasisMultiElectron,
    # but I think it's unnatural and not necessary
    assert basis.is_phonon
    return transform_op_boson(op, basis, boson_encoding)


def get_elem_qubit_op_direct(row_idx: int, col_idx: int, dof: Any):
    if (row_idx, col_idx) == (0, 0):
        return 1 / 2 * (Op("I", dof) + Op("Z", dof))
    elif (row_idx, col_idx) == (0, 1):
        return 1 / 2 * (Op("X", dof) + 1j * Op("Y", dof))
    elif (row_idx, col_idx) == (1, 0):
        return 1 / 2 * (Op("X", dof) - 1j * Op("Y", dof))
    else:
        assert (row_idx, col_idx) == (1, 1)
        return 1 / 2 * (Op("I", dof) - Op("Z", dof))


def transform_op_direct(op: Op, dof: Any, basis: BasisSet):
    if basis.nbas != 2:
        raise ValueError("Direct encoding only support two level basis")
    mat = basis.op_mat(op)
    ret = OpSum()
    for row_idx, col_idx in zip(*np.nonzero(mat)):
        ret += mat[row_idx, col_idx] * get_elem_qubit_op_direct(row_idx, col_idx, dof)
    return ret.simplify(atol=DISCARD_EPS)


def get_elem_qubit_op_unary(row_idx: int, col_idx: int, new_dofs: List[Any]):
    if row_idx == col_idx:
        dof_list = [new_dofs[row_idx]]
        return 1 / 2 * (Op("I", dof_list) - Op("Z", dof_list))
    else:
        des = 1 / 2 * (Op("X", new_dofs[col_idx]) + 1j * Op("Y", new_dofs[col_idx]))
        cre = 1 / 2 * (Op("X", new_dofs[row_idx]) - 1j * Op("Y", new_dofs[row_idx]))
        # exploit the commutation property for simplification
        if new_dofs[row_idx] < new_dofs[col_idx]:
            return cre * des
        else:
            return des * cre


def transform_op_boson_unary(op: Op, dof: Any, basis: BasisSet):
    new_dofs = [(dof, f"{DOF_SALT}-{i}") for i in range(basis.nbas - 1, -1, -1)]
    mat = basis.op_mat(op)
    ret = OpSum()
    for row_idx, col_idx in zip(*np.nonzero(mat)):
        ret += mat[row_idx, col_idx] * get_elem_qubit_op_unary(row_idx, col_idx, new_dofs)
    return ret.simplify(atol=DISCARD_EPS)


def get_elem_qubit_op_binary(row_idx: int, col_idx: int, new_dofs: List[Any], code_strs: List[str]):
    # gray_code_ints = [int(s, base=2) for s in gray_code_strs]
    n_qubits = len(new_dofs)
    if row_idx == col_idx:
        op_list = []
        for i in range(n_qubits):
            dof = new_dofs[i]
            if code_strs[row_idx][i] == "0":
                new_op = 1 / 2 * (Op("I", dof) + Op("Z", dof))
            else:
                new_op = 1 / 2 * (Op("I", dof) - Op("Z", dof))
            op_list.append(new_op)
        return OpSum.product(op_list)
    else:
        # |code1><code2|
        code1 = code_strs[row_idx]
        code2 = code_strs[col_idx]
        # diff_idx = [i for i in bin(code1 ^ code2)[2:] if i]
        op_list = []
        for i in range(n_qubits):
            dof = new_dofs[i]
            if code1[i] == code2[i]:
                if code1[i] == "0":
                    new_op = 1 / 2 * (Op("I", dof) + Op("Z", dof))
                else:
                    new_op = 1 / 2 * (Op("I", dof) - Op("Z", dof))
            else:
                if code1[i] + code2[i] == "01":
                    new_op = 1 / 2 * (Op("X", dof) + 1j * Op("Y", dof))
                else:
                    new_op = 1 / 2 * (Op("X", dof) - 1j * Op("Y", dof))
            op_list.append(new_op)
        return OpSum.product(op_list)


def transform_op_boson_binary(op: Op, dof: Any, basis: BasisSet, encoding: str):
    n_qubits = (basis.nbas - 1).bit_length()
    new_dofs = [(dof, f"{DOF_SALT}-{i}") for i in range(n_qubits)]
    if encoding == "gray":
        code_strs = get_gray_codes(n_qubits)
    else:
        assert encoding == "binary"
        code_strs = get_binary_codes(n_qubits)

    mat = basis.op_mat(op)
    ret = OpSum()
    for row_idx, col_idx in zip(*np.nonzero(mat)):
        ret += mat[row_idx, col_idx] * get_elem_qubit_op_binary(row_idx, col_idx, new_dofs, code_strs)
    return ret.simplify(atol=DISCARD_EPS)


def transform_op_boson(op, basis, encoding=None):
    assert op.factor == 1

    if encoding is None:
        return transform_op_direct(op, op.dofs[0], basis)
    elif encoding == "unary":
        return transform_op_boson_unary(op, op.dofs[0], basis)
    elif encoding.lower() in ["binary", "gray"]:
        return transform_op_boson_binary(op, op.dofs[0], basis, encoding.lower())
    else:
        raise ValueError(f"Encoding '{encoding}' not supported")


def get_gray_codes(n):
    """Return n-bit Gray code in a list."""
    if n == 0:
        return [""]
    sub_gray_codes = get_gray_codes(n - 1)

    gray_codes0 = ["0" + code for code in sub_gray_codes]
    gray_codes1 = ["1" + code for code in reversed(sub_gray_codes)]

    return gray_codes0 + gray_codes1


def get_binary_codes(n):
    codes = [bin(i)[2:] for i in range(1 << n)]
    return ["0" * (n - len(code)) + code for code in codes]


def get_encoding(m, boson_encoding):
    if boson_encoding is None:
        assert m == 2
        encoding_order = "01"
    elif boson_encoding == "unary":
        encoding_order = ["0" * (m - 1 - i) + "1" + "0" * i for i in range(m)]
    elif boson_encoding == "binary":
        encoding_order = get_binary_codes((m - 1).bit_length())[:m]
    else:
        assert boson_encoding == "gray"
        encoding_order = get_gray_codes((m - 1).bit_length())[:m]
    return encoding_order


def get_init_circuit(model_ref, model, boson_encoding, init_condition):
    for k, v in init_condition.items():
        basis = model_ref.dof_to_basis[k]
        if not isinstance(v, int):
            if (
                isinstance(basis, BasisHalfSpin)
                and v.shape == (2, 2)
                and np.allclose(np.eye(2), v @ v.T.conj)
                and np.allclose(np.eye(2), v.T.conj @ v)
            ):
                continue
            else:
                return get_init_circuit_general(model_ref, model, boson_encoding, init_condition)

    # replace the dof_name key to site_index key
    circuit = tc.Circuit(len(model.basis))
    for k, v in init_condition.items():
        basis = model_ref.dof_to_basis[k]
        if isinstance(basis, BasisHalfSpin):
            if v == 1:
                circuit.X(model.dof_to_siteidx[k])
            elif v.shape == (2, 2):
                circuit.ANY(idx, unitary=v)
            else:
                assert v == 0
        elif isinstance(basis, BasisMultiElectron):
            if v == 1:
                idx = model.dof_to_siteidx[basis.dofs]
                circuit.X(idx)
            else:
                assert v == 0
        else:
            assert basis.is_phonon
            if boson_encoding is None:
                assert v in [0, 1]
                target = [v]
            elif boson_encoding == "unary":
                target = [0] * len(basis.nbas)
                target[len(basis.nbas) - v - 1] = 1
            elif boson_encoding == "binary":
                target = get_binary_codes((basis.nbas - 1).bit_length())[v]
            else:
                assert boson_encoding == "gray"
                target = get_gray_codes((basis.nbas - 1).bit_length())[v]
            for i, t in enumerate(target):
                if t == 1:
                    circuit.X(model.dof_to_siteidx[(k, f"{DOF_SALT}-{i}")])
    return circuit


def get_init_circuit_general(model_ref, model, boson_encoding, init_condition):
    mps = Mps.hartree_product_state(model_ref, init_condition)
    mps_state = mps.todense()
    subspace_idx = get_subspace_idx(model_ref.basis, boson_encoding)
    assert len(mps_state) == len(subspace_idx)
    n_qubits = len(model.basis)
    state = np.zeros(1 << n_qubits)
    state[subspace_idx] = mps_state
    return tc.Circuit(n_qubits, inputs=state)


def get_subspace_idx(basis_list, boson_encoding):
    subspace_idx = [""]
    for basis in basis_list:
        if isinstance(basis, (BasisSimpleElectron, BasisMultiElectron, BasisHalfSpin)):
            new_idx = "01"
        else:
            new_idx = get_encoding(basis.nbas, boson_encoding)
        new_subspace_idx = []
        for idx1 in subspace_idx:
            for idx2 in new_idx:
                new_subspace_idx.append(idx1 + idx2)
        subspace_idx = new_subspace_idx
    return [int(i, base=2) for i in subspace_idx]
