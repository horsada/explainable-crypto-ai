from setuptools import setup, find_packages

setup(
    name="explainable_crypto_ai",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
