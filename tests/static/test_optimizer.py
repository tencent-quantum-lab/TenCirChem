#  Copyright (c) 2024. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.

import numpy as np
from scipy.optimize import minimize

from tencirchem import UCCSD
from tencirchem.molecule import h4
from tencirchem.utils.optimizer import soap


def test_optimizer():
    ucc = UCCSD(h4)
    ucc.kernel()

    opt_res = minimize(ucc.energy, ucc.init_guess, method=soap)
    assert np.allclose(opt_res.fun, ucc.e_ucc)
