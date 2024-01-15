import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="tencirchem",
    version="2024.01",
    author="TenCirChem Authors",
    author_email="liw31@163.com",
    description="Efficient quantum computational chemistry based on TensorCircuit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tencent-quantum-lab/tencirchem",
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "tensorcircuit",
        "pyscf",
        "openfermion",
        "qiskit",
        "pylatexenc",
        "noisyopt",
        "renormalizer",
    ],
    extras_require={
        "jax": ["jax", "jaxlib"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
