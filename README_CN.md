# TenCirChem

![TenCirChem](https://github.com/tencent-quantum-lab/TenCirChem/blob/master/docs/source/statics/logov0.png)

[![ci](https://img.shields.io/github/actions/workflow/status/tencent-quantum-lab/tencirchem/ci.yml?branch=master)](https://github.com/tencent-quantum-lab/TenCirChem/actions)
[![codecov](https://codecov.io/github/tencent-quantum-lab/TenCirChem/branch/master/graph/badge.svg?token=6QZP1RKVTT)](https://app.codecov.io/github/tencent-quantum-lab/TenCirChem)
[![pypi](https://img.shields.io/pypi/v/tencirchem.svg?logo=pypi)](https://pypi.org/project/tencirchem/)
[![doc](https://img.shields.io/badge/docs-link-green.svg)](https://tencent-quantum-lab.github.io/TenCirChem/index.html)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tencent-quantum-lab/TenCirChem/master?labpath=docs%2Fsource%2Ftutorial_jupyter)

[English](https://github.com/tencent-quantum-lab/TenCirChem/) | 简体中文

TenCirChem 是一个高效且通用的分子性质量子计算软件包。TenCirChem 基于[TensorCircuit]，并针对化学应用进行了优化。

## 安装
安装 TenCirChem 非常简单。只需通过 pip 安装包即可：

```sh
pip install tencirchem
```

## 简单易用
TenCirChem 用纯 Python 编写，使用起来非常简单。以下是计算 UCCSD 的示例：

```python
from tencirchem import UCCSD, M

d = 0.8
# 距离单位是埃
h4 = M(atom=[["H", 0, 0, d * i] for i in range(4)])

# 配置
uccsd = UCCSD(h4)
# 计算并返回能量
uccsd.kernel()
# 分析结果
uccsd.print_summary(include_circuit=True)
```
在上面的代码中运行 uccsd.kernel() 将确定优化电路ansatz参数和 VQE 能量。
TenCirChem 还允许用户提供自定义参数。以下是示例：

```python
import numpy as np

from tencirchem import UCCSD
from tencirchem.molecule import h4

uccsd = UCCSD(h4)
# 基于自定义参数计算各种属性
params = np.zeros(uccsd.n_params)
print(uccsd.statevector(params))
print(uccsd.energy(params))
print(uccsd.energy_and_grad(params))
```
有关更多示例和自定义，请参见[文档](https://tencent-quantum-lab.github.io/TenCirChem/index.html) 


## 令人兴奋的功能
TenCirChem 的功能包括：
- 静态模块
  - 以极快的速度计算 UCCSD、kUpCCGSD、pUCCD
  - 通过 TensorCircuit 进行噪声电路模拟
  - 自定义积分、活性空间近似、RDM、GPU 支持等。
- 动态模块
  - 将 [renormalizer](https://github.com/shuaigroup/Renormalizer) 模型转换为量子位表示
  - 基于 JAX 的 VQA 算法
  - 内置模型：自旋玻色模型、吡嗪 S1/S2 内部转换动力学


## 设计原则
TenCirChem的设计原则是：
- 快速
  - UCC的速度比其他软件包快10000倍
    - 例如：16个量子比特的H8（CPU）在2秒内完成，20个量子比特的H10（GPU）在14秒内完成
    - 通过UCC因子的解析展开和对称性实现
- 易于修改
  - 尽可能避免定义新类和包装器
    - 例如：激发算符表示为 `int` 的 `tuple`。算符池只是一个 `tuple` 的 `list`
  - 最小的类继承层次结构：至多两层
  - 通过类属性公开内部变量

## 许可证
TenCirChem 使用 Academic Public License 发布。
有关详细信息，请参见[LICENSE 文件](https://github.com/tencent-quantum-lab/TenCirChem/blob/master/LICENSE) 。
简而言之，您可以免费使用 TenCirChem 用于非商业/学术目的，
商业用途需要获得商业许可。

## 引用 TenCirChem
如果这个项目对你的研究有帮助，请引用我们的软件白皮书：

[TenCirChem: An Efficient Quantum Computational Chemistry Package for the NISQ Era](https://arxiv.org/abs/2303.10825)

这也是软件的一个很好的介绍。

## 研究和应用

### 变分基矢编码器
变分基矢编码器是编码电声子系统中的声子用以量子计算的高效算法。
请参考[例子](https://github.com/tencent-quantum-lab/TenCirChem/tree/master/example).
参考论文: https://arxiv.org/pdf/2301.01442.pdf (发表于PRR).

