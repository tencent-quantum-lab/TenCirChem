# TenCirChem

![TenCirChem](https://github.com/tencent-quantum-lab/TenCirChem/blob/master/docs/source/statics/logov0.png)

[![ci](https://img.shields.io/github/actions/workflow/status/tencent-quantum-lab/tencirchem/ci.yml?branch=master)](https://github.com/tencent-quantum-lab/TenCirChem/actions)
[![codecov](https://codecov.io/github/tencent-quantum-lab/TenCirChem/branch/master/graph/badge.svg?token=6QZP1RKVTT)](https://app.codecov.io/github/tencent-quantum-lab/TenCirChem)
[![doc](https://img.shields.io/badge/docs-link-green.svg)](https://tencent-quantum-lab.github.io/TenCirChem/index.html)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tencent-quantum-lab/TenCirChem/master?labpath=docs%2Fsource%2Ftutorial_jupyter)

TenCirChem is an efficient and versatile quantum computation package for molecular properties.
TenCirChem is based on [TensorCircuit](https://github.com/tencent-quantum-lab/tensorcircuit), with heavy optimization for chemistry applications.

## Install
The package is purely written in Python and can be obtained via `pip` as:

```sh
pip install tencirchem
```

## Getting Started
UCCSD calculation example

```python
from tencirchem import UCCSD, M

d = 0.8
# distance unit is angstrom
h4 = M(atom=[["H", 0, 0, d * i] for i in range(4)])

# setup
uccsd = UCCSD(h4)
# calculate
uccsd.kernel()
# analyze result
uccsd.print_summary(include_circuit=True)
```
Plugin your own code is easy

```python
import numpy as np

from tencirchem import UCCSD
from tencirchem.molecule import h4

uccsd = UCCSD(h4)
# evaluate various properties based on custom parameters
params = np.zeros(uccsd.n_params)
print(uccsd.statevector(params))
print(uccsd.energy(params))
print(uccsd.energy_and_grad(params))
```
Please refer to the [documentation](https://tencent-quantum-lab.github.io/TenCirChem/index.html) 
for more examples and customization.

## Features
- Static module
  - Extremely fast UCC calculation with UCCSD, kUpCCGSD, pUCCD
  - Noisy circuit simulation via TensorCircuit
  - Custom integrals, active space approximation, RDMs, GPU support, etc.
- Dynamic module
  - Transformation from [renormalizer](https://github.com/shuaigroup/Renormalizer) models to qubit representation
  - VQA algorithm based on JAX
  - Built-in models: spin-boson model, pyrazine S1/S2 internal conversion dynamics


## Design principle
- Fast
  - UCC speed 10000x faster than other packages
    - Example: H8 with 16 qubits in 2s (CPU). H10 with 20 qubits in 14s (GPU)
    - Achieved by analytical expansion of UCC factors and exploitation of symmetry
- Easy to hack
  - Avoid defining new classes and wrappers when possible
    - Example: Excitation operators are represented as `tuple` of `int`. An operator pool is simply a `list` of `tuple`
  - Minimal class inheritance hierarchy: at most two levels
  - Expose internal variables through class attributes

## License
TenCirChem uses its own [license](https://github.com/tencent-quantum-lab/TenCirChem/blob/master/LICENSE)
adopted from [openCARP](https://opencarp.org/download/license).
In short, you can use TenCirChem freely for non-commercial/academic purpose 
and commercial use requires a commercial license.