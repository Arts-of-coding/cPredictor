[![PyPI version](https://badge.fury.io/py/cPredictor.svg)](https://badge.fury.io/py/cPredictor)
[![CI/CD](https://github.com/Arts-of-coding/cPredictor/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Arts-of-coding/cPredictor/actions/workflows/ci-cd.yml)
[![Maintainability](https://api.codeclimate.com/v1/badges/598ba117b586183c46a8/maintainability)](https://codeclimate.com/github/Arts-of-coding/cPredictor/maintainability)
[![Test Coverage](https://api.codeclimate.com/v1/badges/598ba117b586183c46a8/test_coverage)](https://codeclimate.com/github/Arts-of-coding/cPredictor/test_coverage)
# cma
This repository defines a command-line tool to predict (cPredictor) datasets according to a cell meta-atlases. At the present time only the meta-atlas for the cornea has been implemented.


Install cPredictor into a conda environment and install with PyPI:
```
$ conda create -n cPredictor python=3.9 pip
$ conda activate cPredictor
$ pip install cPredictor
```
To see what each of the current functions do you can run these commands:
```
$ SVM_performance --help
$ SVM_prediction --help
$ SVM_import --help
$ SVM_pseudobulk --help
```
The documentation will be extended and improved upon in later versions.
