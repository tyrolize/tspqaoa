from setuptools import setup

from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="tspqaoa",
    version="0.0.1",
    description="Implementation of QAOA for Travelling Salesman Problem",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Artemiy Burov",
    author_email="artemiy.burov@outlook.com",
    url="https://github.com/tyrolize/tspqaoa",
    python_requires=">=3, <4",
    packages=["tspqaoa"],
    install_requires=[
        "qiskit==0.29.0",
        "pynauty",
        "qiskit-optimization",
        "pandas",
        "networkx",
        "numpy",
        "pytest",
        "tqdm",
        "cvxgraphalgs",
        "cvxopt",
        "scikit-learn==1.0",
        "notebook",
        "matplotlib",
        "seaborn",
    ],
    zip_safe=True,
)