import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="tencirchem",
    version="2024.11",
    author="TenCirChem Authors",
    author_email="liw31@163.com",
    description="Efficient quantum computational chemistry based on TensorCircuit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tencent-quantum-lab/tencirchem",
    packages=setuptools.find_packages(),
    python_requires=">=3.9",
    include_package_data=True,
    install_requires=[
        "numpy==1.26.4",
        "scipy==1.13.1",
        "pandas==2.2.3",
        "tensorcircuit[qiskit]==0.12.0",
        "pyscf==2.7.0",
        "openfermion==1.6.1",
        "pylatexenc==2.10",
        "noisyopt==0.2.2",
        "renormalizer==0.0.10",
    ],
    extras_require={
        "jax": ["jax", "jaxlib"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
