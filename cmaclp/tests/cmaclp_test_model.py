# Import modules
import argparse
import os
import numpy as np
import pandas as pd
import scanpy as sc
import time as tm
import seaborn as sns
import rpy2.robjects as robjects
import matplotlib
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from scanpy import read_h5ad
from importlib.resources import files
import subprocess

reference = "data/cma_meta_atlas.h5ad"
labels = "data/training_labels_meta.csv"

# Add figure map to the import function
# Change the command-line functions to take them really as arguments + add replicates into the test

def test_SVMrej_performance():

    command_to_be_executed = ['SVM_performance',
                              '--reference_H5AD', str(reference),
                              '--LabelsPath', str(labels)]

    subprocess.run(command_to_be_executed, shell=False, timeout=None,
                   text=True)
    assert os.path.exists(f'figures/SVMrej_cnf_matrix.png') == 1
