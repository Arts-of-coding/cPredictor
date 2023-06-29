[![CI/CD](https://github.com/Arts-of-coding/cmaclp/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Arts-of-coding/cmaclp/actions/workflows/ci-cd.yml)
[![Maintainability](https://api.codeclimate.com/v1/badges/598ba117b586183c46a8/maintainability)](https://codeclimate.com/github/Arts-of-coding/cmaclp/maintainability)
[![Test Coverage](https://api.codeclimate.com/v1/badges/598ba117b586183c46a8/test_coverage)](https://codeclimate.com/github/Arts-of-coding/cmaclp/test_coverage)
# cma
This repository defines a command-line tool to predict (clp) datasets according to a cell meta-atlas (cma). At the present time only the meta-atlas for the cornea has been implemented.

To see what each of the current functions do you can run these commands:

```
$ conda create -n cmaclp python=3.9 pip
$ conda activate cmaclp
$ pip install cmaclp
```
With an activated environment you can then run:
```
$ SVM_performance --help
$ SVM_prediction --help
$ SVM_import --help
```
The documentation will be extended and improved upon in later versions.
