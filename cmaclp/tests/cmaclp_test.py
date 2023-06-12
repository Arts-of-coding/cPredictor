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

def obs_key_wise_subsampling(adata, obs_key, N):
    '''
    Subsample each class to same cell numbers (N). Classes are given by obs_key pointing to categorical in adata.obs.
    '''
    counts = adata.obs[obs_key].value_counts()
    # subsample indices per group defined by obs_key
    indices = [np.random.choice(adata.obs_names[adata.obs[obs_key]==group], size=N) for group in counts.index] #, replace=False
    selection = np.hstack(np.array(indices))
    return adata[selection].copy()


query = "data/small_test.h5ad"

# Subset the meta_atlas to 1500 cells of equal categories
ref = sc.read_h5ad("data/cma_meta_atlas.h5ad")
label_data= pd.read_csv("data/training_labels_meta.csv",sep=',',index_col=0)
label_data=label_data.index.tolist()
ref.obs["labels"]=label_data

# Subset 100 cells times 15 categories
sub=obs_key_wise_subsampling(ref, "labels", 100)
sub.var.index = sub.var["_index"]
del sub.var["_index"]
del sub.raw
del ref
sub.write_h5ad("data/small_cma_meta_atlas.h5ad")

# Redefine label data
sub.obs["labels"].to_csv("data/small_training_labels_meta.csv", index=None,header=True)

# OLD
#adata = sc.read_h5ad("data/cma_meta_atlas.h5ad")
#adata = adata[0:10000, :]
#adata.var.index = adata.var["_index"]
#del adata.var["_index"]
#del adata.raw
#adata.write_h5ad("data/small_cma_meta_atlas.h5ad")

reference = "data/small_cma_meta_atlas.h5ad"
labels = "data/small_training_labels_meta.csv"
outdir = "test_output/"

# Add figure map to the import function
# Change the command-line functions to take them really as arguments + add replicates into the test

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

