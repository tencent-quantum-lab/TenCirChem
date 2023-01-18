from tencirchem import UCCSD, KUPCCGSD, PUCCD, M

# distance unit is angstrom
d = 0.8
# M is borrowed from PySCF. In other words, `from tencirchem import M`
# is equivalent to `from pyscf import M`
m = M(atom=[["H", 0, 0, d * i] for i in range(2)])

# setup
uccsd = UCCSD(m)
# calculate
uccsd.kernel()
# analyze result
uccsd.print_summary(include_circuit=True)

# other classes share the same interface
print(KUPCCGSD(m).kernel())
print(PUCCD(m).kernel())
