from setuptools import find_packages, setup

setup(
    name="torchjd",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
