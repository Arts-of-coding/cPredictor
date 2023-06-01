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

query = "small_test.h5ad"

# Subset the meta_atlas to 1000 cells
adata = sc.read_h5ad("cma_meta_atlas.h5ad")
adata = adata[0:1000,:]
adata.var.index=adata.var["_index"]
del adata.var["_index"]
del adata.raw
adata.write_h5ad("small_cma_meta_atlas.h5ad")

reference="small_cma_meta_atlas.h5ad"
labels = "training_labels_meta.csv"
outdir="test_output/"

def test_SVM_prediction():
    command_to_be_executed = ['SVM_prediction', '--reference_H5AD',  str(reference), '--query_H5AD',  str(query), '--LabelsPathTrain',  str(labels), '--OutputDir', str(outdir)]
    subprocess.run(command_to_be_executed, shell=False, timeout=None, text=True)
    assert os.path.exists(f'{outdir}SVM_Pred_Labels.csv') == 1
