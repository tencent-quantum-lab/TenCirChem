# TenCirChem

![TenCirChem](https://github.com/tencent-quantum-lab/TenCirChem/blob/master/docs/source/statics/logov0.png)

[![ci](https://img.shields.io/github/actions/workflow/status/tencent-quantum-lab/tencirchem/ci.yml?branch=master)](https://github.com/tencent-quantum-lab/TenCirChem/actions)
[![codecov](https://codecov.io/github/tencent-quantum-lab/TenCirChem/branch/master/graph/badge.svg?token=6QZP1RKVTT)](https://app.codecov.io/github/tencent-quantum-lab/TenCirChem)
[![pypi](https://img.shields.io/pypi/v/tencirchem.svg?logo=pypi)](https://pypi.org/project/tencirchem/)
[![doc](https://img.shields.io/badge/docs-link-green.svg)](https://tencent-quantum-lab.github.io/TenCirChem/index.html)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tencent-quantum-lab/TenCirChem/master?labpath=docs%2Fsource%2Ftutorial_jupyter)

English | [简体中文](https://github.com/tencent-quantum-lab/TenCirChem/blob/master/README_CN.md)

TenCirChem is an efficient and versatile quantum computation package for molecular properties.
TenCirChem is based on [TensorCircuit](https://github.com/tencent-quantum-lab/tensorcircuit) 
and is optimized for chemistry applications.

## Easy Installation
Getting started with TenCirChem by installing the package via pip:

```sh
pip install tencirchem
```

## Simple to Use
TenCirChem is written in pure Python, and its use is straightforward. Here's an example of calculating UCCSD:

```python
from tencirchem import UCCSD, M

d = 0.8
# distance unit is angstrom
h4 = M(atom=[["H", 0, 0, d * i] for i in range(4)])

# configuration
uccsd = UCCSD(h4)
# calculate and returns energy
uccsd.kernel()
# analyze result
uccsd.print_summary(include_circuit=True)
```
Running uccsd.kernel() in the above code determines the optimized circuit ansatz parameters and VQE energy.  
TenCirChem also allows the user to supply custom parameters. Here's an example:

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
For more examples and customization,
please refer to the [documentation](https://tencent-quantum-lab.github.io/TenCirChem/index.html) 


## Exciting Features
TenCirChem's features include:
- Statics module
  - UCC calculation with UCCSD, kUpCCGSD, pUCCD at an extremely fast speed
  - Noisy circuit simulation via TensorCircuit
  - Custom integrals, active space approximation, RDMs, GPU support, etc.
- Dynamics module
  - Transformation from [renormalizer](https://github.com/shuaigroup/Renormalizer) models to qubit representation
  - VQA algorithm based on JAX
  - Built-in models: spin-boson model, pyrazine S1/S2 internal conversion dynamics


## Design principle
TenCirChem is designed to be:
- Fast
  - UCC speed is 10000x faster than other packages
    - Example: H8 with 16 qubits in 2s (CPU). H10 with 20 qubits in 14s (GPU)
    - Achieved by analytical expansion of UCC factors and exploitation of symmetry
- Easy to hack
  - Avoid defining new classes and wrappers when possible
    - Example: Excitation operators are represented as `tuple` of `int`. An operator pool is simply a `list` of `tuple`
  - Minimal class inheritance hierarchy: at most two levels
  - Expose internal variables through class attributes

## License
TenCirChem is released under Academic Public License.
See the [LICENSE file](https://github.com/tencent-quantum-lab/TenCirChem/blob/master/LICENSE) for details.
In short, you can use TenCirChem freely for non-commercial/academic purpose
and commercial use requires a commercial license.

## Citing TenCirChem
If this project helps in your research, please cite our software whitepaper:

[TenCirChem: An Efficient Quantum Computational Chemistry Package for the NISQ Era](https://arxiv.org/abs/2303.10825)

which is also a good introduction to the software.

## Research and Applications

### Variational basis state encoder
An efficient algorithm to encode phonon states in electron-phonon systems for quantum computation.
See [examples](https://github.com/tencent-quantum-lab/TenCirChem/tree/master/example)
and the [tutorial](https://tencent-quantum-lab.github.io/TenCirChem/tutorial_jupyter/vbe_tutorial_td.html).
Reference paper: https://arxiv.org/pdf/2301.01442.pdf (published in PRR).
