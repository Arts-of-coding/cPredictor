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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from scanpy import read_h5ad
from importlib_resources import files
import subprocess
import re
import matplotlib.pyplot as plt
from statistics import mean
from scipy.stats import pearsonr
from scipy.stats import spearmanr

query = "data/small_test.h5ad"
reference = "data/small_cma_meta_atlas.h5ad"
labels = "data/small_training_labels_meta.csv"

### END-TO-END
outdir = "test_output_end_to_end/"

def test_SVM_prediction():

    command_to_be_executed = ['SVM_prediction',
                              '--reference_H5AD', str(reference),
                              '--query_H5AD', str(query),
                              '--LabelsPathTrain', str(labels),
                              '--OutputDir', str(outdir)]

    subprocess.run(command_to_be_executed, shell=False, timeout=None,
                   text=True)
    assert os.path.exists(f'{outdir}SVM_Pred_Labels.csv') == 1


def test_SVMrej_prediction():

    command_to_be_executed = ['SVM_prediction',
                              '--reference_H5AD', str(reference),
                              '--query_H5AD', str(query),
                              '--LabelsPathTrain', str(labels),
                              '--OutputDir', str(outdir),
                              '--rejected']

    subprocess.run(command_to_be_executed, shell=False, timeout=None,
                   text=True)
    assert os.path.exists(f'{outdir}SVMrej_Pred_Labels.csv') == 1

    
def test_SVM_import():

    command_to_be_executed = ['SVM_import',
                              '--query_H5AD', str(query),
                              '--OutputDir', str(outdir),
                              '--SVM_type', str("SVM"),
                              '--meta-atlas']

    subprocess.run(command_to_be_executed, shell=False, timeout=None,
                   text=True)
    assert os.path.exists(f'figures/Density_prediction_scores.pdf') == 1
    assert os.path.exists(f'SVM_predicted.h5ad') == 1


def test_SVM_import_plots():

    command_to_be_executed = ['SVM_import',
                              '--query_H5AD', str(query),
                              '--OutputDir', str(outdir),
                              '--SVM_type', str("SVM"),
                              '--replicates', str("time_point"),
                              '--show-bar']

    subprocess.run(command_to_be_executed, shell=False, timeout=None,
                   text=True)
    assert os.path.exists(f'figures/SVM_predicted_bar.pdf') == 1
      #assert os.path.exists(f'figures/umap_SVM_predicted.pdf') == 1