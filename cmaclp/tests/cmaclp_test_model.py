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

print("import performance function")
from cmaclp.SVM_prediction import SVM_performance

reference = "data/cma_meta_atlas.h5ad"
labels = "data/training_labels_meta.csv"
outdir = "test_output/"
cmaclp_version = "0.0.10"

metrics = SVM_performance(reference_H5AD=reference,LabelsPath=labels,OutputDir=outdir)
print(metrics)
