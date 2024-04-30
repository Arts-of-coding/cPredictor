Usage
=====

.. _installation:

Installation
------------

To use cPredictor, we recommend to use Docker directly. There are multiple containers where you can directly query your single-cell object against pretrained models (h5ad file).
Alternatively, you can install cPredictor into a :ref:`Conda` or :ref:`Pip` environment. Using Conda and pip, you need to download the files from meta-atlases separately, see :doc:`instructions`.

Docker
----------------

.. code-block:: console

   $ docker pull artsofcoding/cpredictor:latest
   $ docker tag artsofcoding/cpredictor:latest cpredictor

Pip
----------------
You can first create a virtual pip environment, then you can use pip install for cPredictor.

.. code-block:: console

   (.venv) $ pip install cPredictor

Conda
----------------
If you have not used Bioconda (or miniforge) before, first set up the necessary channels (in this order!). 
You only have to do this once.

.. code-block:: console

   $ conda config --add channels defaults
   $ conda config --add channels bioconda
   $ conda config --add channels conda-forge
   
Then you can create a conda environment and install cPredictor via pip.

.. code-block:: console

   $ conda create -n cPredictor python=3.9 pip
   $ conda activate cPredictor
   (cPredictor) $ pip install cPredictor
