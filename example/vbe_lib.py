#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.

"""
Library for variational basis state encoder
https://arxiv.org/abs/2301.01442
"""
import numpy as np
from opt_einsum import contract


def get_psi_indices(dof_nature, b_dof_pidx, n_qubit_per_mode):
    b_dof_vidx = []
    for i in range(max(dof_nature) + 1):
        b_dof_vidx.append(np.where(dof_nature == i)[0])

    # indices used for tensor contraction
    psi_idx = []
    for i in range(max(dof_nature) + 1 + list(dof_nature).count(-1)):
        if i not in b_dof_pidx:
            psi_idx.extend([f"p-{i}"])
        else:
            psi_idx.extend([f"v-{i}-{j}" for j in range(n_qubit_per_mode)])
    psi_idx_top = [f"{i}-top" for i in psi_idx]
    psi_idx_bottom = [f"{i}-bottom" for i in psi_idx]

    return psi_idx_top, psi_idx_bottom, b_dof_vidx


def get_contracted_mpo(h_mpo, b_array, n_qubit_per_mode, b_dof_pidx, psi_indices):
    nbas = b_array.shape[-1]
    b_shape = tuple([2] * n_qubit_per_mode + [nbas])
    args = []
    for i in range(len(h_mpo)):
        args.append(h_mpo[i])
        args.append([f"mpo-{i}", f"p-{i}-top", f"p-{i}-bottom", f"mpo-{i + 1}"])
    for i in range(len(b_array)):
        dof_pidx = b_dof_pidx[i]
        args.append(b_array[i].reshape(b_shape).conj())
        args.append([f"v-{dof_pidx}-{j}-top" for j in range(n_qubit_per_mode)] + [f"p-{dof_pidx}-top"])
        args.append(b_array[i].reshape(b_shape))
        args.append([f"v-{dof_pidx}-{j}-bottom" for j in range(n_qubit_per_mode)] + [f"p-{dof_pidx}-bottom"])
    out = psi_indices.copy()
    out.extend(["mpo-0", f"mpo-{len(h_mpo)}"])
    args.append(out)
    size = round(2 ** (len(psi_indices) // 2))
    h_contracted = contract(*args).reshape(size, size)
    return h_contracted


def get_contract_args(psi, h_mpo, b_array, i, n_qubit_per_mode, psi_idx_top, psi_idx_bottom, b_dof_pidx):
    nbas = b_array.shape[-1]
    psi_shape2 = [2] * len(psi_idx_top)
    b_shape = tuple([2] * n_qubit_per_mode + [nbas])
    # psi
    args = [psi.reshape(psi_shape2).conj(), psi_idx_top] + [psi.reshape(psi_shape2), psi_idx_bottom]
    # H
    for j in range(len(h_mpo)):
        args.append(h_mpo[j])
        args.append([f"mpo-{j}", f"p-{j}-top", f"p-{j}-bottom", f"mpo-{j + 1}"])
    # b
    for j in range(len(b_array)):
        if j == i:
            continue
        k = b_dof_pidx[j]
        indices = [f"v-{k}-{l}-bottom" for l in range(n_qubit_per_mode)] + [f"p-{k}-bottom"]
        args.extend([b_array[j].reshape(b_shape), indices])
        indices = [f"v-{k}-{l}-top" for l in range(n_qubit_per_mode)] + [f"p-{k}-top"]
        args.extend([b_array[j].reshape(b_shape).conj(), indices])
    return args
