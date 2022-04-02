from setuptools import setup, find_packages


PACKAGENAME = "mpipso"
VERSION = "0.0.dev"


setup(
    name=PACKAGENAME,
    version=VERSION,
    author="Andrew Hearin",
    author_email="ahearin@anl.gov",
    description="Parallel particle swarm",
    long_description="Particle Swarm Optimization with mpi4py",
    install_requires=["numpy", "mpi4py", "jax"],
    packages=find_packages(),
    url="https://github.com/aphearin/mpipso",
)
