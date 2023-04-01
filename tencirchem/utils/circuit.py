#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.


from typing import Tuple

import pandas as pd
from opt_einsum import contract_path
import tensornetwork as tn
import tensorcircuit as tc


def get_circuit_dataframe(circuit: tc.Circuit):
    n_gates = circuit.gate_count()
    n_cnot = circuit.gate_count(["cnot"])
    n_mc = circuit.gate_count(["multicontrol"])
    # avoid int overflow and for better formatting
    n_qubits = circuit.circuit_param["nqubits"]
    flop = count_circuit_flop(circuit)

    df_dict = {
        "#qubits": n_qubits,
        "#gates": [n_gates],
        "#CNOT": [n_cnot],
        "#multicontrol": [n_mc],
        "depth": [circuit.to_qiskit().depth()],
        "#FLOP": [flop],
    }
    return pd.DataFrame(df_dict)


def count_circuit_flop(circuit: tc.Circuit):
    # tensor network contraction flops by greedy algorithm
    nodes = circuit._copy()[0]
    input_set_list = [set([id(e) for e in node.edges]) for node in nodes]
    array_list = [node.tensor for node in nodes]
    output_set = set([id(e) for e in tn.get_subgraph_dangling(nodes)])
    args = []
    for i in range(len(nodes)):
        args.extend([array_list[i], input_set_list[i]])
    args.append(output_set)
    _, desc = contract_path(*args)
    return desc.opt_cost


def evolve_pauli(circuit: tc.Circuit, pauli_string: Tuple, theta: float):
    # pauli_string in openfermion.QubitOperator.terms format
    for idx, symbol in pauli_string:
        if symbol == "X":
            circuit.H(idx)
        elif symbol == "Y":
            circuit.SD(idx)
            circuit.H(idx)
        elif symbol == "Z":
            continue
        else:
            raise ValueError(f"Invalid Pauli String: {pauli_string}")

    for i in range(len(pauli_string) - 1):
        circuit.CNOT(pauli_string[i][0], pauli_string[i + 1][0])
    circuit.rz(pauli_string[-1][0], theta=theta)

    for i in reversed(range(len(pauli_string) - 1)):
        circuit.CNOT(pauli_string[i][0], pauli_string[i + 1][0])

    for idx, symbol in pauli_string:
        if symbol == "X":
            circuit.H(idx)
        elif symbol == "Y":
            circuit.H(idx)
            circuit.S(idx)
        elif symbol == "Z":
            continue
        else:
            raise ValueError(f"Invalid Pauli String: {pauli_string}")
    return circuit


def multicontrol_ry(theta):
    # https://arxiv.org/pdf/2005.14475.pdf
    c = tc.Circuit(4)
    i, j, k, l = 0, 1, 2, 3

    c.x(i)
    c.x(k)

    c.ry(l, theta=theta / 8)
    c.h(k)
    c.cnot(l, k)

    c.ry(l, theta=-theta / 8)
    c.h(i)
    c.cnot(l, i)

    c.ry(l, theta=theta / 8)
    c.cnot(l, k)

    c.ry(l, theta=-theta / 8)
    c.h(j)
    c.cnot(l, j)

    c.ry(l, theta=theta / 8)
    c.cnot(l, k)

    c.ry(l, theta=-theta / 8)
    c.cnot(l, i)

    c.ry(l, theta=theta / 8)
    c.h(i)
    c.cnot(l, k)

    # there's a typo in the paper
    c.ry(l, theta=-theta / 8)
    c.h(k)
    c.cnot(l, j)
    c.h(j)

    c.x(i)
    c.x(k)
    return c
