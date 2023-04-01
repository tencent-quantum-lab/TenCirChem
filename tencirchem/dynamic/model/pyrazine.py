#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.


import logging
from itertools import permutations as permut
from itertools import product


import numpy as np
from renormalizer.model import Op
from renormalizer.model.basis import BasisSHO, BasisMultiElectron
from renormalizer.utils.constant import ev2au


logger = logging.getLogger(__name__)


r"""
Bi-linear vibronic coupling model for Pyrazine, 4 modes
See: Raab, Worth, Meyer, Cederbaum.  J.Chem.Phys. 110 (1999) 936
The parameters are from heidelberg mctdh package pyr4+.op
"""


# frequencies
w10a = 0.1139 * ev2au
w6a = 0.0739 * ev2au
w1 = 0.1258 * ev2au
w9a = 0.1525 * ev2au

# energy-gap
delta = 0.42300 * ev2au

# linear, on-diagonal coupling coefficients
# H(1,1)
_6a_s1_s1 = 0.09806 * ev2au
_1_s1_s1 = 0.05033 * ev2au
_9a_s1_s1 = 0.14521 * ev2au
# H(2,2)
_6a_s2_s2 = -0.13545 * ev2au
_1_s2_s2 = 0.17100 * ev2au
_9a_s2_s2 = 0.03746 * ev2au

# quadratic, on-diagonal coupling coefficients
# H(1,1)
_10a_10a_s1_s1 = -0.01159 * ev2au
_6a_6a_s1_s1 = 0.00000 * ev2au
_1_1_s1_s1 = 0.00000 * ev2au
_9a_9a_s1_s1 = 0.00000 * ev2au
# H(2,2)
_10a_10a_s2_s2 = -0.01159 * ev2au
_6a_6a_s2_s2 = 0.00000 * ev2au
_1_1_s2_s2 = 0.00000 * ev2au
_9a_9a_s2_s2 = 0.00000 * ev2au

# bilinear, on-diagonal coupling coefficients
# H(1,1)
_6a_1_s1_s1 = 0.00108 * ev2au
_1_9a_s1_s1 = -0.00474 * ev2au
_6a_9a_s1_s1 = 0.00204 * ev2au
# H(2,2)
_6a_1_s2_s2 = -0.00298 * ev2au
_1_9a_s2_s2 = -0.00155 * ev2au
_6a_9a_s2_s2 = 0.00189 * ev2au

# linear, off-diagonal coupling coefficients
_10a_s1_s2 = 0.20804 * ev2au

# bilinear, off-diagonal coupling coefficients
# H(1,2) and H(2,1)
_1_10a_s1_s2 = 0.00553 * ev2au
_6a_10a_s1_s2 = 0.01000 * ev2au
_9a_10a_s1_s2 = 0.00126 * ev2au

e_list = ["s1", "s2"]
v_list = ["10a", "6a", "9a", "1"]


def get_ham_terms():
    ham_terms = []
    for e_isymbol, e_jsymbol in product(e_list, repeat=2):
        e_op = Op(r"a^\dagger a", [e_isymbol, e_jsymbol])
        for v_isymbol, v_jsymbol in product(v_list, repeat=2):
            # linear
            if v_isymbol == v_jsymbol:
                # if one of the permutation is defined, then the `e_idx_tuple` term should
                # be defined as required by Hermitian Hamiltonian
                for eterm1, eterm2 in permut([e_isymbol, e_jsymbol], 2):
                    factor = globals().get(f"_{v_isymbol}_{eterm1}_{eterm2}")
                    if factor is not None:
                        factor *= np.sqrt(eval(f"w{v_isymbol}"))
                        ham_terms.append(e_op * Op("x", v_isymbol) * factor)
                        logger.debug(f"term: {v_isymbol}_{e_isymbol}_{e_jsymbol}")
                        break
                else:
                    logger.debug(f"no term: {v_isymbol}_{e_isymbol}_{e_jsymbol}")

            # quadratic
            # use product to guarantee `break` breaks the whole loop
            it = product(permut([v_isymbol, v_jsymbol], 2), permut([e_isymbol, e_jsymbol], 2))
            for (vterm1, vterm2), (eterm1, eterm2) in it:
                factor = globals().get(f"_{vterm1}_{vterm2}_{eterm1}_{eterm2}")

                if factor is not None:
                    factor *= np.sqrt(eval(f"w{v_isymbol}") * eval(f"w{v_jsymbol}"))
                    if vterm1 == vterm2:
                        v_op = Op("x^2", vterm1)
                    else:
                        v_op = Op("x", vterm1) * Op("x", vterm2)
                    ham_terms.append(e_op * v_op * factor)
                    logger.debug(f"term: {v_isymbol}_{v_jsymbol}_{e_isymbol}_{e_jsymbol}")
                    break
            else:
                logger.debug(f"no term: {v_isymbol}_{v_jsymbol}_{e_isymbol}_{e_jsymbol}")

    # electronic coupling
    ham_terms.append(Op(r"a^\dagger a", "s1", -delta, [0, 0]))
    ham_terms.append(Op(r"a^\dagger a", "s2", delta, [0, 0]))

    # vibrational kinetic and potential
    for v_isymbol in v_list:
        ham_terms.extend([Op("p^2", v_isymbol, 0.5), Op("x^2", v_isymbol, 0.5 * eval("w" + v_isymbol) ** 2)])

    ham_terms = [term for term in ham_terms if term.factor != 0]

    return ham_terms


def get_basis(nlevels):
    if isinstance(nlevels, int):
        nlevels = [nlevels] * len(v_list)
    if not isinstance(nlevels, list):
        raise TypeError(f"`nlevels` must be int or list. Got {type(nlevels)}")
    basis = [BasisMultiElectron(e_list, [0, 0])]
    for v_isymbol, n in zip(v_list, nlevels):
        basis.append(BasisSHO(v_isymbol, globals()[f"w{v_isymbol}"], n))
    return basis
