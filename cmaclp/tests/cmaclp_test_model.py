# Import modules
import argparse
import os
import numpy as np
import pandas as pd
import scanpy as sc
import time as tm
import seaborn as sns
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

print("Import performance function")
from cmaclp.SVM_prediction import SVM_performance

reference = "data/cma_meta_atlas.h5ad"
labels = "data/training_labels_meta.csv"
outdir = "test_output/"
cmaclp_version = "0.1.0"

metrics = SVM_performance(reference_H5AD=reference,LabelsPath=labels,OutputDir=outdir)

print("Setup ML tracking packages")
import dagshub
import mlflow

print("Setup environments and tokens")
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import git

# Set environments and passwords
DAGSHUB_USER_NAME = 'Arts-of-coding'
DAGSHUB_TOKEN =  os.environ['DH_key']
DAGSHUB_REPO_OWNER = 'Arts-of-coding'
MLFLOW_EXPERIMENT_NAME = 'Cornea'
DAGSHUB_REPO_NAME='cmaclp'
Upload_type="GHA"
SVM_model="SVMrej"

os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USER_NAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN
os.environ['MLFLOW_TRACKING_URI'] = f'https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow'
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
os.environ['MLFLOW_EXPERIMENT_NAME'] = MLFLOW_EXPERIMENT_NAME

print("Upload metrics to dagshub for model tracking")

# Create new experiment if it does not exist yet otherwise add to the current experiment
try:
    experiment_id = mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
except:
    pass

mlflow.start_run(run_name=str(f'{MLFLOW_EXPERIMENT_NAME}_full_{Upload_type}'),experiment_id=experiment_id,
                 tags={"version": str(cmaclp_version),"model": str(SVM_model)})
mlflow.log_metric("weighted_F1_score", metrics[0])
mlflow.log_metric("weighted_accuracy_score", metrics[1])
mlflow.log_metric("weighted_precision_score", metrics[2])

mlflow.end_run()
