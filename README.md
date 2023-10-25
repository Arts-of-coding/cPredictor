# `cPredictor` package <img src='cPredictor/man/c_predictor_simple.png' align="right" height="80" />

[![PyPI version](https://badge.fury.io/py/cPredictor.svg)](https://badge.fury.io/py/cPredictor)
[![CI/CD](https://github.com/Arts-of-coding/cPredictor/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Arts-of-coding/cPredictor/actions/workflows/ci-cd.yml)
[![Maintainability](https://api.codeclimate.com/v1/badges/598ba117b586183c46a8/maintainability)](https://codeclimate.com/github/Arts-of-coding/cPredictor/maintainability)
[![Test Coverage](https://api.codeclimate.com/v1/badges/598ba117b586183c46a8/test_coverage)](https://codeclimate.com/github/Arts-of-coding/cPredictor/test_coverage)
[<img src="https://img.shields.io/badge/dockerhub-images-blue.svg?logo=Docker">](https://hub.docker.com/repository/docker/artsofcoding/cpredictor/general)

This repository defines a command-line tool to predict (cPredictor) datasets according to a cell meta-atlases. At the present time only the meta-atlas for the cornea has been implemented.

## Conda and pip
If you have not used Bioconda before, first set up the necessary channels (in this order!). 
You only have to do this once.
```
$ conda config --add channels defaults
$ conda config --add channels bioconda
$ conda config --add channels conda-forge
```

Install cPredictor into a conda environment and install with PyPI:
```
$ conda create -n cPredictor python=3.9 pip
$ conda activate cPredictor
$ pip install cPredictor
```
To see what each of the current functions do you can run these commands:
```
$ SVM_performance --help
$ SVM_predict --help
$ SVM_import --help
$ SVM_pseudobulk --help
```
## Docker
Alternatively you can run the package containerized through docker:
```
$ docker pull artsofcoding/cpredictor:latest
$ docker tag artsofcoding/cpredictor:latest cpredictor
$ docker run -it --name cpredictor -p 8080:80 -v {path_to_H5AD_object}:/data cpredictor
```
In the activated docker container you can then go to the terminal:
```
# cd /data
# SVM_predict --query_H5AD {H5AD_object}.h5ad --OutputDir {your_output_dir} --meta_atlas
```
## Performance with the corneal meta-atlas
The docker container is able to predict the identity of ~90.000 cells x ~25.000 genes within two hours.

To run the container locally you will need a computer with at least 28 GB of RAM and a 4-core processor.

The documentation will be extended and improved upon in later versions.
