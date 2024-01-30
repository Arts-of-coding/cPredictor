# Import modules
import argparse
import os
import numpy as np
import pandas as pd
import scanpy as sc
import time as tm
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from scanpy import read_h5ad
from importlib_resources import files
import subprocess
import re
import matplotlib.pyplot as plt
from statistics import mean

# Import standalone functions for unit tests
from cPredictor.SVM_prediction import SVM_predict
from cPredictor.SVM_prediction import SVM_import
from cPredictor.SVM_prediction import SVM_pseudobulk
from cPredictor.SVM_prediction import SVM_performance
from cPredictor.SVM_prediction import predpars
from cPredictor.SVM_prediction import importpars
from cPredictor.SVM_prediction import performpars

query = "data/small_test.h5ad"

reference = "data/cma_meta_atlas.h5ad"
labels = "data/training_labels_meta.csv"
outdir_unit = "test_output_unit/"
colord_tsv= "data/colord.tsv"

### UNIT TESTS

def test_SVM_predict():
    SVM_predict(reference_H5AD=reference,query_H5AD=query,
      LabelsPath=labels,OutputDir=outdir_unit)

    assert os.path.exists(f'{outdir_unit}SVM_Pred_Labels.csv') == 1


def test_SVM_predict_meta_atlas():
    SVM_predict(query_H5AD=query,meta_atlas=True,
      OutputDir=outdir_unit)

    assert os.path.exists(f'{outdir_unit}SVM_Pred_Labels.csv') == 1


def test_SVMrej_predict():
    SVM_predict(reference_H5AD=reference,query_H5AD=query,
      LabelsPath=labels,OutputDir=outdir_unit,rejected=True)

    assert os.path.exists(f'{outdir_unit}SVMrej_Pred_Labels.csv') == 1


def test_SVM_predict_meta_atlas():
    SVM_predict(reference_H5AD=reference,query_H5AD=query,
      LabelsPath=labels,OutputDir=outdir_unit,rejected=True)

    assert os.path.exists(f'{outdir_unit}SVMrej_Pred_Labels.csv') == 1

    
def test_SVM_import():

    SVM_import(query_H5AD=query,OutputDir=outdir_unit,SVM_type="SVM",replicates="time_point")

    assert os.path.exists(f'figures/Density_prediction_scores.pdf') == 1
    assert os.path.exists(f'SVM_predicted.h5ad') == 1


def test_SVM_plot_import():

    SVM_import(query_H5AD=query,OutputDir=outdir_unit,
      SVM_type="SVM",replicates="time_point",show_bar=True)

    assert os.path.exists(f'figures/SVM_predicted_bar.pdf') == 1

def test_SVMrej_plot_import():

    SVM_import(query_H5AD=query,OutputDir=outdir_unit,
      SVM_type="SVMrej",replicates="time_point",show_bar=True)

    assert os.path.exists(f'figures/SVMrej_predicted_bar.pdf') == 1

def test_SVM_plot_import_meta_atlas():

    SVM_import(query_H5AD=query,OutputDir=outdir_unit,colord=colord_tsv,
      SVM_type="SVM",replicates="time_point",show_bar=True,meta_atlas=True)

    assert os.path.exists(f'figures/SVM_predicted_bar.pdf') == 1
      
def test_SVM_pseudobulk():

    SVM_pseudobulk(condition_1=reference, condition_1_batch="donors", 
      condition_2="SVM_predicted.h5ad", condition_2_batch="batch", Labels_1=labels)
    assert os.path.exists("pseudobulk_output/full_batch_samples.tsv") == 1
    assert os.path.exists("pseudobulk_output/merged_batch_samples.tsv") == 1

# Issue to still fix, no priority for now
#def test_SVMrej_pseudobulk():

#    SVM_pseudobulk(condition_1=reference, condition_1_batch="donors", 
#      condition_2="SVMrej_predicted.h5ad", condition_2_batch="batch", 
#      Labels_1=labels,SVM_type="SVMrej")
#    assert os.path.exists("pseudobulk_output/full_batch_samples.tsv") == 1
#    assert os.path.exists("pseudobulk_output/merged_batch_samples.tsv") == 1

def test_SVM_performance():

    SVM_performance(reference_H5AD=reference,LabelsPath=labels,OutputDir=outdir_unit, rejected=False)
    assert os.path.exists("figures/SVM_cnf_matrix.png") == 1
    
def test_SVMrej_performance():

    SVM_performance(reference_H5AD=reference,LabelsPath=labels,OutputDir=outdir_unit, rejected=True)
    assert os.path.exists("figures/SVMrej_cnf_matrix.png") == 1

### END-TO-END
outdir = "test_output_end_to_end/"

def test_CLI_SVM_predict():

    command_to_be_executed = ['SVM_predict',
                              '--reference_H5AD', str(reference),
                              '--query_H5AD', str(query),
                              '--LabelsPath', str(labels),
                              '--OutputDir', str(outdir)]

    subprocess.run(command_to_be_executed, shell=False, timeout=None,
                   text=True)
    assert os.path.exists(f'{outdir}SVM_Pred_Labels.csv') == 1


def test_CLI_SVMrej_predict():

    command_to_be_executed = ['SVM_predict',
                              '--reference_H5AD', str(reference),
                              '--query_H5AD', str(query),
                              '--LabelsPath', str(labels),
                              '--OutputDir', str(outdir),
                              '--rejected']

    subprocess.run(command_to_be_executed, shell=False, timeout=None,
                   text=True)
    assert os.path.exists(f'{outdir}SVMrej_Pred_Labels.csv') == 1

    
def test_CLI_SVM_import():

    command_to_be_executed = ['SVM_import',
                              '--query_H5AD', str(query),
                              '--OutputDir', str(outdir),
                              '--SVM_type', str("SVM"),
                              '--meta-atlas']

    subprocess.run(command_to_be_executed, shell=False, timeout=None,
                   text=True)
    assert os.path.exists(f'figures/Density_prediction_scores.pdf') == 1
    assert os.path.exists(f'SVM_predicted.h5ad') == 1


def test_CLI_SVM_import_plots():

    command_to_be_executed = ['SVM_import',
                              '--query_H5AD', str(query),
                              '--OutputDir', str(outdir),
                              '--SVM_type', str("SVM"),
                              '--replicates', str("time_point"),
                              '--show-bar']

    subprocess.run(command_to_be_executed, shell=False, timeout=None,
                   text=True)
    assert os.path.exists(f'figures/SVM_predicted_bar.pdf') == 1

    
def test_CLI_SVM_pseudobulk():

    command_to_be_executed = ['SVM_pseudobulk',
                              '--condition_1', str(reference),
                              '--condition_1_batch', str("donors"),
                              '--condition_2', str("SVM_predicted.h5ad"),
                              '--condition_2_batch', str("batch"),
                              '--Labels_1', str(labels)]

    subprocess.run(command_to_be_executed, shell=False, timeout=None,
                   text=True)
    assert os.path.exists("pseudobulk_output/full_batch_samples.tsv") == 1
    assert os.path.exists("pseudobulk_output/merged_batch_samples.tsv") == 1
