Installation Instructions
=========================

Installation
------------
``pip install multiswarm``

Prerequisites
-------------
- Tested on Python versions: ``python=3.9-12``
- MPI (several implementations available - see https://pypi.org/project/mpi4py)

Example installation with conda env:
++++++++++++++++++++++++++++++++++++

.. code-block:: bash

    conda create -n py312 python=3.12
    conda activate py312
    conda install -c conda-forge mpi4py
    pip install multiswarm
