#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.

import numpy as np

a = np.array([[0, 1], [0, 0]])
# a^\dagger
ad = a.T
# a^\dagger_i a_j
ad_a = np.kron(ad, a)
# a^\dagger_i a_j - a_j a^\dagger_i
ad_a_hc = ad_a - ad_a.T
adad_aa = np.kron(np.kron(np.kron(ad, ad), a), a)
adad_aa_hc = adad_aa - adad_aa.T
ad_a_hc2 = ad_a_hc @ ad_a_hc
adad_aa_hc2 = adad_aa_hc @ adad_aa_hc

DISCARD_EPS = 1e-12
