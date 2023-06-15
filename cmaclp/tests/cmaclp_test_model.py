# Import modules
import argparse
import os
import numpy as np
import pandas as pd
import scanpy as sc
import time as tm
import seaborn as sns
import cmaclp
from sklearn.svm import LinearSVC
import rpy2.robjects as robjects
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from scanpy import read_h5ad
from importlib_resources import files
import subprocess

reference = "data/cma_meta_atlas.h5ad"
labels = "data/training_labels_meta.csv"
outdir = "test_output/"

# Add figure map to the import function
# Change the command-line functions to take them really as arguments + add replicates into the test

def test_SVMrej_performance():

    command_to_be_executed = ['SVM_performance',
                              '--reference_H5AD', str(reference),
                              '--LabelsPath', str(labels),
                              '--OutputDir', str(outdir)]

    subprocess.run(command_to_be_executed, shell=False, timeout=None,
                   text=True)
    assert os.path.exists(f'figures/SVMrej_cnf_matrix.png') == 1