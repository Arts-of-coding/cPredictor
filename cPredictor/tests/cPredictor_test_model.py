# Import modules
import argparse
import gc
import os
import numpy as np
import pandas as pd
import pyarrow as pa
import scanpy
import time as tm
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.svm import LinearSVC
from sklearn.calibration import (CalibratedClassifierCV, calibration_curve, CalibrationDisplay)
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (train_test_split, StratifiedKFold, KFold)
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler)
from sklearn.metrics import (confusion_matrix, f1_score, accuracy_score, precision_score)
import logging
import pickle
import joblib
import json
from importlib.resources import files
from scanpy import read_h5ad
import subprocess

# Importing custom functions/modules if any
from cPredictor.SVM_prediction import (CpredictorClassifier,CpredictorClassifierPerformance)

# Add module and check if it works, if it does work, move it to processing scripts
class NaivelyCalibratedLinearSVC(LinearSVC):
    """LinearSVC with `predict_proba` method that naively scales
    `decision_function` output for binary classification."""

    def fit(self, X, y):
        super().fit(X, y)
        df = self.decision_function(X)
        self.df_min_ = df.min()
        self.df_max_ = df.max()

    def predict_proba(self, X):
        """Min-max scale output of `decision_function` to [0, 1]."""
        df = self.decision_function(X)
        calibrated_df = (df - self.df_min_) / (self.df_max_ - self.df_min_)
        proba_pos_class = np.clip(calibrated_df, 0, 1)
        proba_neg_class = 1 - proba_pos_class
        proba = np.c_[proba_neg_class, proba_pos_class]
        return proba

def SVM_calibration(reference_H5AD, LabelsPath, OutputDir, rejected=True, Threshold_rej=0.7):
    '''
    Tests performance of SVM model based on a reference H5AD dataset.

    Parameters:
    reference_H5AD : H5AD file of datasets of interest.
    OutputDir : Output directory defining the path of the exported SVM_predictions.
    SVM_type: Type of SVM prediction, SVM or SVMrej (default).
    '''
    logging.basicConfig(level=logging.DEBUG, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S',
                        filename='cPredictor_performance.log', filemode='w')

    # Using the child class of the CpredictorClassifier to process the data
    cpredictorperf = CpredictorClassifierPerformance(Threshold_rej, rejected, OutputDir)

    svc = NaivelyCalibratedLinearSVC(max_iter=10_000, dual="auto")
    svc_isotonic = CalibratedClassifierCV(svc, cv=3, method="isotonic")
    svc_sigmoid = CalibratedClassifierCV(svc, cv=3, method="sigmoid")

    # Setup cpredictors params from its class
    kf = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 42)
    cpredictormodel = getattr(cpredictorperf,"Classifier")
    svc_cpredictor = CalibratedClassifierCV(cpredictormodel, cv=kf)

    clf_list = [
        (svc, "SVC"),
        (svc_isotonic, "SVC + Isotonic"),
        (svc_sigmoid, "SVC + Sigmoid"),
        (svc_cpredictor, "SVC + cPredictor"),
    ]

    logging.info('Reading in the data')
    Data = read_h5ad(reference_H5AD)
    
    Data = cpredictorperf.expression_cutoff(Data, LabelsPath)

    data_train = pd.DataFrame.sparse.from_spmatrix(Data.X, index=list(Data.obs.index.values), columns=list(Data.var.index.values))
    data_train = data_train.to_numpy(dtype="float16")
    
    data_train = cpredictorperf.preprocess_data_train(data_train)
    #data_train_processed = data_train
    
    # Do label encoding
    labels = pd.read_csv(LabelsPath, header=0,index_col=None, sep=',') #, usecols = col
    label_encoder = LabelEncoder()
    
    y = label_encoder.fit_transform(labels.iloc[:,0].tolist())

    # Generate a dictionary to map values to strings
    res = dict(zip(label_encoder.inverse_transform(y),y))
    res['Unlabeled'] = 999999
    res = {v: k for k, v in res.items()}
    res

    # CV 3
    y_binaries = []
    for cls in range(len(np.unique(y))):
        y_binary = np.where(y == cls, 1, 0)
        y_binaries.append(y_binary)
    print(y_binaries)

    for i, y_binary in enumerate(y_binaries):
        name_cond=''.join(list(label_encoder.inverse_transform([i])))
        print(name_cond)
        X_train, X_test, y_train, y_test = train_test_split(data_train, y_binaries[i], test_size=0.2, random_state=42)
        colors = plt.get_cmap("Dark2")
        fig = plt.figure(figsize=(5, 8))
        gs = gridspec.GridSpec(4, 2)

        ax_calibration_curve = fig.add_subplot(gs[:2, :2])
        calibration_displays = {}
        for i, (clf, name) in enumerate(clf_list):
            clf.fit(X_train, y_train)
            display = CalibrationDisplay.from_estimator(
                clf,
                X_test,
                y_test,
                n_bins=10,
                name=name,
                ax=ax_calibration_curve,
                color=colors(i),
                pos_label=1
            )
            calibration_displays[name] = display

        ax_calibration_curve.grid()
        ax_calibration_curve.set_title(f"Calibration plots (SVC) {name_cond}")

        # Add histogram
        grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
        for i, (_, name) in enumerate(clf_list):
            row, col = grid_positions[i]
            ax = fig.add_subplot(gs[row, col])

            ax.hist(
                calibration_displays[name].y_prob,
                range=(0, 1),
                bins=10,
                label=name,
                color=colors(i),
            )
            ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

        plt.tight_layout()
        #plt.savefig(f"{name_cond}_calibrated.png")
        mlflow.log_figure(fig,f"{name_cond}_calibrated.png")
        #plt.show()
    return

# Setup the scripts
svc = NaivelyCalibratedLinearSVC(max_iter=10_000, dual="auto")
svc_isotonic = CalibratedClassifierCV(svc, cv=3, method="isotonic")
svc_sigmoid = CalibratedClassifierCV(svc, cv=3, method="sigmoid")

clf_list = [
    (svc, "SVC"),
    (svc_isotonic, "SVC + Isotonic"),
    (svc_sigmoid, "SVC + Sigmoid"),
]

os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import git

# Specify data types and output dirs
reference = "test/cma_meta_atlas_rfe.h5ad"
labels = "data/training_labels_meta.csv"
outdir = "test_output/"
cPredictor_version = "0.3.5"

print("Setup tokens")
# Set environments and passwords
DAGSHUB_USER_NAME = 'Arts-of-coding'
DAGSHUB_TOKEN =  os.environ['DH_key']
DAGSHUB_REPO_OWNER = 'Arts-of-coding'
MLFLOW_EXPERIMENT_NAME = 'Cornea'
DAGSHUB_REPO_NAME='cPredictor'
Upload_type="GHA"
SVM_model="SVMrej"

print("Setup ML tracking packages")
import dagshub
import mlflow

os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USER_NAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN
os.environ['MLFLOW_TRACKING_URI'] = f'https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow'
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
os.environ['MLFLOW_EXPERIMENT_NAME'] = MLFLOW_EXPERIMENT_NAME

print("Upload metrics to dagshub for model tracking")

# Read in metrics from the produced file of the CLI command
with open(f"{outdir}metrics.txt", "r") as file:
    metrics = []
    for line in file:
        metric = float(line)
        metrics.append(metric)

# Create new experiment if it does not exist yet otherwise add to the current experiment
try:
    expid = mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
except:
    print(f'Using existing run of {MLFLOW_EXPERIMENT_NAME}')
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

mlflow.start_run(run_name=str(f'{MLFLOW_EXPERIMENT_NAME}_full_{Upload_type}'),experiment_id=None,
                 tags={"version": str(cPredictor_version),"model": str(SVM_model)})
mlflow.log_metric("weighted_F1_score", metrics[0])
mlflow.log_metric("weighted_accuracy_score", metrics[1])
mlflow.log_metric("weighted_precision_score", metrics[2])

# Run calibration function
SVM_calibration(reference_H5AD=reference,LabelsPath=labels,OutputDir=outdir, rejected=False)

mlflow.end_run()
