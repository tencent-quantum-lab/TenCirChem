#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.


from renormalizer import BasisHalfSpin, BasisSHO, Op


def get_ham_terms(epsilon, delta, nmode, omega_list, g_list):
    terms = [Op("sigma_z", "spin", epsilon), Op("sigma_x", "spin", delta)]
    for i in range(nmode):
        g = g_list[i]
        omega = omega_list[i]
        terms.extend([Op(r"b^\dagger b", f"v{i}", omega), Op("sigma_z", "spin", g) * Op(r"b^\dagger+b", f"v{i}")])
    return terms


def get_basis(omega_list, nlevels):
    if isinstance(nlevels, int):
        nlevels = [nlevels] * len(omega_list)
    if not isinstance(nlevels, list):
        raise TypeError(f"`nlevels` must be int or list. Got {type(nlevels)}")
    basis = [BasisHalfSpin("spin")]
    for i in range(len(omega_list)):
        basis.append(BasisSHO(f"v{i}", omega=omega_list[i], nbas=nlevels[i]))
    return basis
