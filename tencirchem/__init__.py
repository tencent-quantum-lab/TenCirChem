__version__ = "2023.03"
__author__ = "TenCirChem Authors"
__creator__ = "Weitang Li"

#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.

import os
import logging

os.environ["JAX_ENABLE_X64"] = "True"
# for debugging
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# disable CUDA 11.1 warning
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logger = logging.getLogger("tensorcircuit")
logger.setLevel(logging.FATAL)

os.environ["RENO_LOG_LEVEL"] = "100"

logger = logging.getLogger("tencirchem")
logger.setLevel(logging.WARNING)

# finish logger stuff
del logger

from tencirchem.utils.backend import set_backend, set_dtype

# by default use float64 rather than float32
set_dtype("complex128")

# static module
# as an external interface
from pyscf import M

from tencirchem.static.ucc import UCC
from tencirchem.static.uccsd import UCCSD
from tencirchem.static.kupccgsd import KUPCCGSD
from tencirchem.static.puccd import PUCCD
from tencirchem.static.hea import HEA, parity, binary, get_noise_conf

# dynamic module
# as an external interface
from renormalizer import Op, BasisSHO, BasisHalfSpin, BasisSimpleElectron, BasisMultiElectron, Model, Mpo
from renormalizer.model import OpSum

from tencirchem.utils.misc import get_dense_operator
from tencirchem.dynamic.time_evolution import TimeEvolution


def clear_cache():
    from tencirchem.utils.backend import ALL_JIT_LIBS
    from .static.evolve_civector import CI_OPERATOR_CACHE, CI_OPERATOR_BATCH_CACHE

    for l in ALL_JIT_LIBS:
        l.clear()
    CI_OPERATOR_CACHE.clear()
    CI_OPERATOR_BATCH_CACHE.clear()
