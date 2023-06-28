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

# Import standalone functions for unit tests
from cmaclp.SVM_prediction import SVM_prediction
from cmaclp.SVM_prediction import SVM_import
from cmaclp.SVM_prediction import SVM_pseudobulk

# OLD
#def obs_key_wise_subsampling(adata, obs_key, N):
#    '''
#    Subsample each class to same cell numbers (N). Classes are given by obs_key pointing to categorical in adata.obs.
#    '''
#    counts = adata.obs[obs_key].value_counts()
#    # subsample indices per group defined by obs_key
#    indices = [np.random.choice(adata.obs_names[adata.obs[obs_key]==group], size=N) for group in counts.index] #, replace=False
#    selection = np.hstack(np.array(indices))
#    return adata[selection].copy()


query = "data/small_test.h5ad"

# OLD
# Subset the meta_atlas to 1500 cells of equal categories
#ref = sc.read_h5ad("data/cma_meta_atlas.h5ad")
#label_data= pd.read_csv("data/training_labels_meta.csv",sep=',',index_col=0)
#label_data=label_data.index.tolist()
#ref.obs["labels"]=label_data

# Subset 100 cells times 15 categories
#sub=obs_key_wise_subsampling(ref, "labels", 100)
#del sub.raw
#del ref
#sub.write_h5ad("data/small_cma_meta_atlas.h5ad")

# Redefine label data
#sub.obs["labels"].to_csv("data/small_training_labels_meta.csv", index=None,header=True)

reference = "data/small_cma_meta_atlas.h5ad"
labels = "data/small_training_labels_meta.csv"
outdir_unit = "test_output_unit/"

### UNIT TESTS

def test_unit_SVM_prediction():

    SVM_prediction(reference_H5AD=reference,query_H5AD=query,
      LabelsPathTrain=labels,OutputDir=outdir_unit)

    assert os.path.exists(f'{outdir_unit}SVM_Pred_Labels.csv') == 1


def test_unit_SVMrej_prediction():

    SVM_prediction(reference_H5AD=reference,query_H5AD=query,
      LabelsPathTrain=labels,OutputDir=outdir_unit,rejected=True)

    assert os.path.exists(f'{outdir_unit}SVMrej_Pred_Labels.csv') == 1

    
def test_unit_SVM_import():

    SVM_import(query_H5AD=query,OutputDir=outdir_unit,SVM_type="SVM",replicates="time_point",meta_atlas=True)

    assert os.path.exists(f'figures/Density_prediction_scores.pdf') == 1
    assert os.path.exists(f'SVM_predicted.h5ad') == 1


def test_unit_SVM_import_plots():

    SVM_import(query_H5AD=query,OutputDir=outdir_unit,
      SVM_type="SVM",replicates="time_point",show_bar=True)

    assert os.path.exists(f'figures/SVM_predicted_bar.pdf') == 1

      
def test_unit_SVM_pseudobulk():

    SVM_pseudobulk(condition_1=reference, condition_1_batch="donors", 
      condition_2="SVM_predicted.h5ad", condition_2_batch="batch", Labels_1=labels)
    assert os.path.exists("pseudobulk_output/full_batch_samples.tsv") == 1
    assert os.path.exists("pseudobulk_output/merged_batch_samples.tsv") == 1

