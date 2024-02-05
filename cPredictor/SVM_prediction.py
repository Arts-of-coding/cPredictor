import argparse
import gc
import os
import numpy as np
import pandas as pd
import scanpy as sc
import time as tm
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearnex import patch_sklearn 
patch_sklearn()
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import pyarrow as pa
from scanpy import read_h5ad
import logging

class CpredictorClassifier():
    def __init__(self, Threshold_rej, rejected, OutputDir):
        self.scaler = MinMaxScaler()
        self.Classifier = LinearSVC(dual = False, random_state = 42, class_weight = 'balanced', max_iter = 2500)
        self.threshold = Threshold_rej
        self.rejected = rejected
        self.output_dir = OutputDir
        self.expression_treshold = 162

    def expression_cutoff(self, Data, LabelsPath):
        logging.info(f'Selecting genes based on an summed expression threshold of minimally {self.expression_treshold} in each cluster')
        labels = pd.read_csv(LabelsPath,index_col=False)
        h5ad_object = Data.copy()
        cluster_id = 'labels'
        h5ad_object.obs[cluster_id] = labels.iloc[:, 0].tolist()
        res = pd.DataFrame(columns=h5ad_object.var_names.tolist(), index=h5ad_object.obs[cluster_id].astype("category").unique())
        
        ## Set up scanpy object based on expression treshold
        for clust in h5ad_object.obs[cluster_id].astype("category").unique():
            if h5ad_object.raw is not None:
                res.loc[clust] = h5ad_object[h5ad_object.obs[cluster_id].isin([clust]),:].raw.X.sum(0)
            else:
                res.loc[clust] = h5ad_object[h5ad_object.obs[cluster_id].isin([clust]),:].X.sum(0)
        res.loc["sum"]=np.sum(res,axis=0).tolist()
        res=res.transpose()
        res=res.loc[res['sum'] > self.expression_treshold]
        genes_expressed = res.index.tolist()
        logging.info("Amount of genes that remain: " + str(len(genes_expressed)))
        h5ad_object = h5ad_object[:, genes_expressed]
        Data = h5ad_object
        del res, h5ad_object

        return Data
        
    def preprocess_data_train(self, data_train):
        logging.info('Log normalizing the training data')
        np.log1p(data_train, out=data_train)
        logging.info('Scaling the training data')
        data_train = self.scaler.fit_transform(data_train)
        return data_train

    def preprocess_data_test(self, data_test):
        logging.info('Log normalizing the testing data')
        np.log1p(data_test, out=data_test)
        logging.info('Scaling the testing data')
        data_test = self.scaler.fit_transform(data_test)
        return data_test

    def fit_and_predict_svmrejection(self, labels_train, threshold, output_dir, data_train, data_test):
        self.rejected = True
        self.threshold = threshold
        self.output_dir = output_dir
        logging.info('Running SVMrejection')
        kf = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 42)
        clf = CalibratedClassifierCV(self.Classifier, cv=kf)
        clf.fit(data_train, labels_train.ravel())
        predicted = clf.predict(data_test)
        prob = np.max(clf.predict_proba(data_test), axis = 1)
        unlabeled = np.where(prob < self.threshold)

        # For unlabeled values from the SVMrejection put values of strings and integers
        try:
            predicted[unlabeled] = 'Unlabeled'
        except ValueError:
            unlabeled = list(unlabeled[0])
            predicted[unlabeled] = 999999
        self.predictions = predicted
        self.probabilities = prob
        self.save_results(self.rejected)

    def fit_and_predict_svm(self, labels_train, output_dir, data_train, data_test):
        self.rejected = False
        self.output_dir = output_dir
        logging.info('Running SVM')
        self.Classifier.fit(data_train, labels_train.ravel())
        self.predictions = self.Classifier.predict(data_test)
        self.save_results(self.rejected)

    def save_results(self, rejected):
        self.rejected = rejected
        self.predictions = pd.DataFrame(self.predictions)
        if self.rejected is True:
            self.probabilities = pd.DataFrame(self.probabilities)
            self.predictions.to_csv(f"{self.output_dir}/SVMrej_Pred_Labels.csv", index=False)
            self.probabilities.to_csv(f"{self.output_dir}/SVMrej_Prob.csv", index=False)
        else:
            self.predictions.to_csv(f"{self.output_dir}/SVM_Pred_Labels.csv", index=False)

# Child class for performance from the CpredictorClassifier class        
class CpredictorClassifierPerformance(CpredictorClassifier):
    def __init__(self, Threshold_rej, rejected, OutputDir):
        super().__init__(Threshold_rej, rejected, OutputDir)

    def fit_and_predict_svmrejection(self, labels_train, threshold, output_dir, data_train, data_test):
        # Calls the function from parent class and extends it for the child
        super().fit_and_predict_svmrejection(labels_train, threshold, output_dir, data_train, data_test)
        return self.predictions, self.probabilities

    def fit_and_predict_svm(self, labels_train, OutputDir, data_train, data_test):
        # Calls the function from parent class and extends it for the child
        super().fit_and_predict_svm(labels_train, OutputDir, data_train, data_test)
        return self.predictions

def SVM_predict(reference_H5AD, query_H5AD, LabelsPath, OutputDir, rejected=False, Threshold_rej=0.7,meta_atlas=False):
    '''
    run baseline classifier: SVM
    Wrapper script to run an SVM classifier with a linear kernel on a benchmark dataset with 5-fold cross validation,
    outputs lists of true and predicted cell labels as csv files, as well as computation time.

    Parameters:
    reference_H5AD, query_H5AD : H5AD files that produce training and testing data,
        cells-genes matrix with cell unique barcodes as row names and gene names as column names.
    LabelsPath : Cell population annotations file path matching the training data (.csv).
    OutputDir : Output directory defining the path of the exported file.
    rejected: If the flag is added, then the SVMrejected option is chosen. Default: False.
    Threshold_rej : Threshold used when rejecting the cells, default is 0.7.
    meta_atlas : If the flag is added the predictions will use meta-atlas data.
    meaning that reference_H5AD and LabelsPath do not need to be specified.
    '''
    logging.basicConfig(level=logging.DEBUG, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S',
                        filename='cPredictor_predict.log', filemode='w')
    logging.info('Reading in the reference and query H5AD objects')
    
    # Load in the cma.h5ad object or use a different reference
    if meta_atlas is False:
        training = read_h5ad(reference_H5AD) 
    if meta_atlas is True:
        meta_dir = 'data/cma_meta_atlas.h5ad'
        training = read_h5ad(meta_dir) 

    # Get an instance of the Cpredictor class
    cpredictor = CpredictorClassifier(Threshold_rej, rejected, OutputDir)
    training = cpredictor.expression_cutoff(training, LabelsPath)
    
    # Load in the test data
    testing = read_h5ad(query_H5AD)

    # Checks if the test data contains a raw data slot and sets it as the count value
    try:
        testing = testing.raw.to_adata()
    except AttributeError:
        logging.warning('Query object does not contain raw data, using sparce matrix from adata.X')
        logging.warning('Please manually check if this sparce matrix contains actual raw counts')

    logging.info('Generating training and testing matrices from the H5AD objects')
    
    # training data
    matrix_train = pd.DataFrame.sparse.from_spmatrix(training.X, index=list(training.obs.index.values), columns=list(training.var.features.values))

    # testing data
    try: 
        testing.var['features']
    except KeyError:
        testing.var['features'] = testing.var.index
        logging.debug('Setting the var index as var features')
    
    matrix_test = pd.DataFrame.sparse.from_spmatrix(testing.X, index=list(testing.obs.index.values), columns=list(testing.var.features.values))
    
    logging.info('Unifying training and testing matrices for shared genes')
    
    # subselect the train matrix for values that are present in both
    df_all = training.var[["features"]].merge(testing.var[["features"]].drop_duplicates(), on=['features'], how='left', indicator=True)
    df_all['_merge'] == 'left_only'
    training1 = df_all[df_all['_merge'] == 'both']
    col_one_list = training1['features'].tolist()

    matrix_test = matrix_test[matrix_test.columns.intersection(col_one_list)]
    matrix_train = matrix_train[matrix_train.columns.intersection(col_one_list)]
    matrix_train = matrix_train[list(matrix_test.columns)]
    
    logging.info('Number of genes remaining after unifying training and testing matrices: '+str(len(matrix_test.columns)))
    
    # Convert the ordered dataframes back to nparrays
    data_train = matrix_train.to_numpy(dtype="float16")
    data_test = matrix_test.to_numpy(dtype="float16")

    # Save test data on-disk for efficient memory data management
    data_test = pa.Table.from_pandas(pd.DataFrame(data_test))
    with pa.OSFile('data_test.arrow', 'wb') as sink, pa.RecordBatchFileWriter(sink, data_test.schema) as writer:
    	writer.write_table(data_test)
    
    # Delete large objects from memory
    del matrix_train, matrix_test, training, testing, data_test, training1, df_all

    # Run garbage collector
    gc.collect()

    # If meta_atlas=True it will read the training_labels
    if meta_atlas is True:
        LabelsPath = 'data/training_labels_meta.csv'
    
    labels_train = pd.read_csv(LabelsPath, header=0,index_col=None, sep=',')
    labels_train = labels_train.values
    
    # Load in the test data from on disk incrementally
    with pa.memory_map('data_test.arrow', 'rb') as source:
        data_test = pa.ipc.open_file(source).read_all()
        data_test = data_test.to_pandas().to_numpy()

    # Running cpredictor classifier
    logging.info('Running cPredictor classifier')
    data_train = cpredictor.preprocess_data_train(data_train)
    data_test = cpredictor.preprocess_data_test(data_test)
    
    if rejected is True:
        cpredictor.fit_and_predict_svmrejection(labels_train, Threshold_rej, OutputDir, data_train, data_test)
        cpredictor.save_results(rejected)
        
    else:
        cpredictor.fit_and_predict_svm(labels_train, OutputDir, data_train, data_test)
        cpredictor.save_results(rejected)


def SVM_import(query_H5AD, OutputDir, SVM_type, replicates, colord=None, meta_atlas=False, show_bar=False):
    '''
    Imports the output of the SVM_predictor and saves it to the query_H5AD.

    Parameters:
    query_H5AD: H5AD file of datasets of interest.
    OutputDir: Output directory defining the path of the exported SVM_predictions.
    SVM_type: Type of SVM prediction, SVM (default) or SVMrej.
    Replicates: A string value specifying a column in query_H5AD.obs.
    colord: A .tsv file with the order of the meta-atlas and corresponding colors.
    meta_atlas : If the flag is added the predictions will use meta-atlas data.
    show_bar: Shows bar plots depending on the SVM_type, split over replicates.

    '''
    logging.basicConfig(level=logging.DEBUG, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S',
                        filename='cPredictor_import.log', filemode='w')
    logging.info('Reading query data')
    
    adata = read_h5ad(query_H5AD)
    SVM_key = f"{SVM_type}_predicted"

    # Load in the object and add the predicted labels
    logging.info('Adding predictions to query data')
    for file in os.listdir(OutputDir):
        if file.endswith('.csv'):
            if 'rej' not in file:
                filedir = OutputDir+file
                SVM_output_dir = pd.read_csv(filedir,sep=',',index_col=0)
                SVM_output_dir = SVM_output_dir.index.tolist()
                adata.obs["SVM_predicted"] = SVM_output_dir
            if 'rej_Pred' in file:
                filedir = OutputDir+file
                SVM_output_dir = pd.read_csv(filedir,sep=',',index_col=0)
                SVM_output_dir = SVM_output_dir.index.tolist()
                adata.obs["SVMrej_predicted"] = SVM_output_dir
            if 'rej_Prob' in file:
                filedir = OutputDir+file
                SVM_output_dir = pd.read_csv(filedir,sep=',',index_col=0)
                SVM_output_dir = SVM_output_dir.index.tolist()
                adata.obs["SVMrej_predicted_prob"]=SVM_output_dir

    # Set category colors:
    if meta_atlas is True and colord is not None:
        df_category_colors = pd.read_csv(colord, header=None,index_col=False, sep='\t')
        df_category_colors.columns = ['Category', 'Color']
        category_colors = dict(zip(df_category_colors['Category'], df_category_colors['Color']))
        if SVM_key == "SVMrej_predicted":
          category_colors["Unlabeled"] = "#808080"
                    
    if meta_atlas is False or colord is None:
    
      # Load a large color palette
      palette_name = "tab20"
      cmap = plt.get_cmap(palette_name)
      palette = [matplotlib.colors.rgb2hex(c) for c in cmap.colors] 
      
      # Extract the list of colors
      colors = palette
      key_cats = adata.obs[SVM_key].astype("category")
      key_list = key_cats.cat.categories.to_list()
            
      category_colors = dict(zip(key_list, colors[:len(key_list)]))
            
    # Plot absolute and relative barcharts across replicates
    logging.info('Plotting barcharts')
    if show_bar is True:
        sc.set_figure_params(figsize=(15, 5))
        
        key = SVM_key
        obs_1 = key
        obs_2 = replicates

        #n_categories = {x : len(adata.obs[x].cat.categories) for x in [obs_1, obs_2]}
        adata.obs[obs_1] = adata.obs[obs_1].astype("category")
        adata.obs[obs_2] = adata.obs[obs_2].astype("category")
        df = adata.obs[[obs_2, obs_1]].values

        obs2_clusters = adata.obs[obs_2].cat.categories.tolist()
        obs1_clusters = adata.obs[obs_1].cat.categories.tolist()
        obs1_to_obs2 = {k: np.zeros(len(obs2_clusters), dtype="i")
                           for k in obs1_clusters}
        obs2_to_obs1 = {k: np.zeros(len(obs1_clusters), dtype="i")
                           for k in obs2_clusters}
        obs2_to_obs1

        for b, v in df:
            obs2_to_obs1[b][obs1_clusters.index(str(v))] += 1
            obs1_to_obs2[v][obs2_clusters.index(str(b))] += 1

        df = pd.DataFrame.from_dict(obs2_to_obs1,orient = 'index').reset_index()
        df = df.set_index(["index"])
        df.columns = obs1_clusters
        df.index.names = ['Replicate']

        if meta_atlas is True and colord is not None:
            palette = category_colors
            if SVM_type == 'SVM' :
              ord_list = list(palette.keys())
              
            if SVM_type == 'SVMrej':
              ord_list = list(palette.keys())
            
            # Sorts the df on the longer ordered list
            def sort_small_list(long_list, small_list):
              sorted_list = sorted(small_list, key=lambda x: long_list.index(x))
              return sorted_list
            
            sorter = sort_small_list(ord_list, df.columns.tolist())
            
            # Retrieve the color codes from the sorted list
            lstval = [palette[key] for key in sorter]
            
            try:
              df = df[sorter]
            except KeyError:
              df = df
        else:
            lstval = list(category_colors.values())

        stacked_data = df.apply(lambda x: x*100/sum(x), axis=1)
        stacked_data = stacked_data.iloc[:, ::-1]

        fig, ax = plt.subplots(1,2)
        df.plot(kind="bar", stacked=True, ax=ax[0], legend = False,color=lstval, rot=45, title='Absolute number of cells')

        fig.legend(loc=7,title="Cell state")

        stacked_data.plot(kind="bar", stacked=True, legend = False, ax=ax[1],color=lstval[::-1], rot=45, title='Percentage of cells')

        fig.tight_layout()
        fig.subplots_adjust(right=0.9)
        fig.savefig(f"figures/{SVM_key}_bar.pdf", bbox_inches='tight')
        plt.close(fig)
    else:
        None

    if SVM_key == "SVM_predicted":
        logging.info('Plotting label prediction certainty scores')
        category_colors = category_colors

        # Create a figure and axes
        fig, ax = plt.subplots(1,1)

        # Iterate over each category and plot the density
        for category, color in category_colors.items():
            subset = adata.obs[adata.obs['SVM_predicted'] == category]
            sns.kdeplot(data=subset['SVMrej_predicted_prob'], fill=True, color=color, label=f"{category} (Median: {subset['SVMrej_predicted_prob'].median():.2f})", ax=ax)

        # Set labels and title
        plt.xlabel('SVM Certainty Scores')
        plt.ylabel('Density')
        plt.title('Stacked Density Plots of Prediction Certainty Scores by Cell State')

        # Add a legend
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

        # Saving the density plot
        fig.savefig("figures/Density_prediction_scores.pdf", bbox_inches='tight')
        plt.close(fig)
    else:
        None
        
    logging.info('Saving H5AD file')
    adata.write_h5ad(f"{SVM_key}.h5ad")
    return

def SVM_performance(reference_H5AD, LabelsPath, OutputDir, rejected=True, Threshold_rej=0.7, fold_splits=5):
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

    logging.info('Reading in the data')
    Data = read_h5ad(reference_H5AD)

    # Using the child class of the CpredictorClassifier to process the data
    cpredictorperf = CpredictorClassifierPerformance(Threshold_rej, rejected, OutputDir)
    
    Data = cpredictorperf.expression_cutoff(Data, LabelsPath)

    data_train = pd.DataFrame.sparse.from_spmatrix(Data.X, index=list(Data.obs.index.values), columns=list(Data.var.index.values))
    data_train = data_train.to_numpy(dtype="float16")
    
    data_train = cpredictorperf.preprocess_data_train(data_train)
    data_train_processed = data_train
    
    # Do label encoding
    labels = pd.read_csv(LabelsPath, header=0,index_col=None, sep=',') #, usecols = col
    label_encoder = LabelEncoder()
    
    y = label_encoder.fit_transform(labels.iloc[:,0].tolist())

    # Generate a dictionary to map values to strings
    res = dict(zip(label_encoder.inverse_transform(y),y))
    res['Unlabeled'] = 999999
    res = {v: k for k, v in res.items()}
    res

    # Perform cross-validation using train_test_split
    kfold = KFold(n_splits=fold_splits, shuffle=True, random_state=42)

    train_indices = []
    test_indices = []

    # Iterate over each fold and split the data
    logging.info('Generate indices for train and test')
    for train_index, test_index in kfold.split(data_train_processed):
        labels_train, y_test = y[train_index], y[test_index]

        # Store the indices of the training and test sets for each fold
        train_indices.append(list(train_index))
        test_indices.append(list(test_index))

    # Run the SVM model
    test_ind = test_indices
    train_ind = train_indices
    #if SVM_type == "SVMrej":
    #    clf = CalibratedClassifierCV(Classifier, cv=3)

    tr_time=[]
    ts_time=[]
    truelab = []
    pred = []
    prob_full = []

    for i in range(fold_splits):
        logging.info(f"Running cross-val {i}")
        data_train = data_train_processed[train_ind[i]]
        data_test = data_train_processed[test_ind[i]]
        labels_train = y[train_ind[i]]
        y_test = y[test_ind[i]]

        if rejected is True:
            start = tm.time()
            SVM_type = "SVMrej"
            predicted, prob = cpredictorperf.fit_and_predict_svmrejection(labels_train, 
                                                                          Threshold_rej, 
                                                                          OutputDir, 
                                                                          data_train, 
                                                                          data_test)
            pred.extend(predicted.iloc[:, 0].tolist())
            prob_full.extend(prob.iloc[:, 0].tolist())
            ts_time.append(tm.time()-start)

        if rejected is False:
            start = tm.time()
            SVM_type = "SVM"
            predicted = cpredictorperf.fit_and_predict_svm(labels_train, OutputDir, 
                                                                            data_train, data_test)
            pred.extend(predicted.iloc[:, 0].tolist())
            ts_time.append(tm.time()-start)

        truelab.extend(y_test)

    truelab = pd.DataFrame(truelab)
    pred = pd.DataFrame(pred)

# Check truelab and predicted
    
    tr_time = pd.DataFrame(tr_time)
    ts_time = pd.DataFrame(ts_time)

    # Calculating the weighted F1 score:
    F1score = f1_score(truelab[0].to_list(), pred[0].to_list(), average='weighted')
    logging.info(f"The {SVM_type} model ran with the weighted F1 score of: {F1score}")

    # Calculating the weighted accuracy score:
    acc_score = accuracy_score(truelab[0].to_list(), pred[0].to_list())
    logging.info(f"The {SVM_type} model ran with the weighted accuracy score of: {acc_score}")

    # Calculating the weighted precision score:
    prec_score = precision_score(truelab[0].to_list(),  pred[0].to_list(),average="weighted")
    logging.info(f"The {SVM_type} model ran with the weighted precision score of: {prec_score}")

    # Relabel truelab and predicted values by names
    truelab[0] = truelab[0].replace(res)
    pred[0] = pred[0].replace(res)

    logging.info('Saving labels to specified output directory')
    truelab.to_csv(f"{OutputDir}{SVM_type}_True_Labels.csv", index = False)
    pred.to_csv(f"{OutputDir}{SVM_type}_Pred_Labels.csv", index = False)
    tr_time.to_csv(f"{OutputDir}{SVM_type}_Training_Time.csv", index = False)
    ts_time.to_csv(f"{OutputDir}{SVM_type}_Testing_Time.csv", index = False)

    ## Plot the SVM figures
    logging.info('Plotting the confusion matrix')
    true = pd.read_csv(f"{OutputDir}{SVM_type}_True_Labels.csv")
    pred = pd.read_csv(f"{OutputDir}{SVM_type}_Pred_Labels.csv")

    true.columns = ["True"]
    pred.columns = ["Predicted"]

    # Set labels
    labels = list(set(np.hstack((true["True"].astype(str).unique(), pred["Predicted"].astype(str).unique()))))

    # Create confusion matrix
    cnf_matrix = pd.DataFrame(confusion_matrix(true["True"], pred["Predicted"], labels=labels), index=labels, columns=labels)
    cnf_matrix = cnf_matrix.loc[true["True"].astype(str).unique(), pred["Predicted"].astype(str).unique()]
    cnf_matrix = cnf_matrix.div(cnf_matrix.sum(1), axis=0)

    cnf_matrix = cnf_matrix.sort_index(axis=1)
    cnf_matrix = cnf_matrix.sort_index()

    # Plot png
    sns.set(font_scale=0.8)
    cm = sns.clustermap(cnf_matrix.T, cmap="Blues", annot=True,fmt='.2%', row_cluster=False,col_cluster=False)
    cm.savefig(f"figures/{SVM_type}_cnf_matrix.png")
    return F1score,acc_score,prec_score

def SVM_pseudobulk(condition_1, condition_1_batch, condition_2, condition_2_batch, Labels_1, OutputDir="pseudobulk_output/", min_cells=50, SVM_type="SVM"):
    '''
    Produces pseudobulk RNA-seq count files and sample files of either technical or biological replicates.
    It produces pseudobulk of all labels from LabelsPath against predicted labels after SVMprediction.
    Moreover, it produces overall pseudobulk of condition_1 vs condition_2 split by indicated batches.

    Parameters:
    condition_1, condition_2 : H5AD files with cells-genes matrix with cell unique barcodes as 
        row names and gene names as column names. Condition_1 should be the meta-atlas and
        condition_2 should be the queried object.
    condition_1_batch : batch name for the meta-atlas object replicates (biological, technical etc.)
    condition_2_batch: batch name for the queried object
    Labels_1 : Cell population annotations file path matching the training data (.csv) or 
        a string that specifies an .obs value in condition 1.
    OutputDir: The directory into which the results are outputted; default: "pseudobulk_output/"
    '''
    logging.basicConfig(level=logging.DEBUG, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S',
                        filename='cPredictor_pseudobulk.log', filemode='w')
    
    logging.info('Reading in the reference and query H5AD objects and adding batches')
    cond_1 = read_h5ad(condition_1)
    cond_2 = read_h5ad(condition_2)
    outputdir = OutputDir
    
    # Reading in the Labels_1 for the replicates
    # First tries to read in Labels_1 as a path
    try:
      label_data = pd.read_csv(Labels_1,sep=',',index_col=0)
      label_data = label_data.index.tolist()
      cond_1.obs["meta_atlas"] = label_data
      cond_1_label="meta_atlas"

    # Then tries to read in Labels_1 as a string in obs
    except (TypeError, FileNotFoundError):
      try:
        cond_1_label = str(Labels_1)
        cond_1.obs[cond_1_label]

    # If no key if found a valid key must be entered
      except KeyError:
        raise ValueError('Please provide a valid name for labels in Labels_1')
      
    logging.info('Constructing condition for condition 1')
    cond_1.obs["condition"] = "cond1"

    logging.info('Constructing batches for condition 1')

    # Add extra index to the ref object:
    cond_1.obs["batch"] = cond_1.obs[condition_1_batch]

    # Convert the 'Category' column to numeric labels
    cond_1.obs["batch"] = pd.factorize(cond_1.obs["batch"])[0] + 1

    cond_1.obs["merged"] = cond_1.obs[cond_1_label]+"."+cond_1.obs["condition"]
    cond_1.obs["merged_batch"] = cond_1.obs["merged"]+"."+cond_1.obs["batch"].astype(str)
    cond_1.obs["full_batch"] = cond_1.obs["condition"]+"."+cond_1.obs["batch"].astype(str)

    logging.info('Constructing batches for condition 2')

    # Uses the specified SVM_type for predicted cond_2
    cond_2_label = f'{SVM_type}_predicted'
    batch = condition_2_batch

    cond_2.obs["condition"] = "cond2"
    cond_2.obs["batch"] = cond_2.obs[batch]
    cond_2.obs["merged"] = cond_2.obs[cond_2_label].astype(str)+"."+cond_2.obs["condition"]
    cond_2.obs["merged_batch"] = cond_2.obs["merged"]+"."+cond_2.obs["batch"].astype(str)
    cond_2.obs["full_batch"] = cond_2.obs["condition"]+"."+cond_2.obs["batch"].astype(str)
    
    os.makedirs(outputdir, exist_ok=True)
        
    # Use each object as a condition
    for cond in cond_1,cond_2:
        cond_str = cond.obs["condition"].astype(str)[1]
        logging.info(f"Running with {cond_str}")

        if cond.obs["condition"].astype(str)[1] == "cond1":
            cond_name = "cond1"
        elif cond.obs["condition"].astype(str)[1] == "cond2":
            cond_name = "cond2"

        # Construct the sample.tsv file    
        cond.obs["assembly"]="hg38"

        adata = cond.copy()

        # Extract names of genes
        try:
            adata.var_names = adata.var["_index"].tolist()
        except KeyError:
            adata.var_names = adata.var.index


        for cluster_id in ["merged_batch","full_batch"]:
            logging.info("Running pseudobulk extraction with: "+str(cluster_id))
            if cluster_id == "full_batch":
                sample_lists=adata.obs[[cluster_id,"assembly","condition","full_batch"]]
            if cluster_id == "merged_batch":
                sample_lists=adata.obs[[cluster_id,"assembly","merged","condition","full_batch"]]  
            sample_lists = sample_lists.reset_index(drop=True)
            sample_lists = sample_lists.drop_duplicates()
            sample_lists.rename(columns={ sample_lists.columns[0]: "sample" }, inplace = True)

            rna_count_lists = []
            FPKM_count_lists = []
            cluster_names = []

            for cluster in adata.obs[cluster_id].astype("category").unique():

                # Only use ANANSE on clusters with more than minimal amount of cells
                n_cells = adata.obs[cluster_id].value_counts()[cluster]

                if n_cells > min_cells:
                    cluster_names.append(str(cluster))

                    # Generate the raw count file
                    adata_sel = adata[adata.obs[cluster_id].isin([cluster])].copy()
                    adata_sel.raw = adata_sel

                    logging.info(
                        str("gather data from " + cluster + " with " + str(n_cells) + " cells")
                    )

                    X_clone = adata_sel.X.tocsc()
                    X_clone.data = np.ones(X_clone.data.shape)
                    NumNonZeroElementsByColumn = X_clone.sum(0)
                    rna_count_lists += [list(np.array(NumNonZeroElementsByColumn)[0])]
                    sc.pp.normalize_total(adata_sel, target_sum=1e6, inplace=True)
                    X_clone2 = adata_sel.X.toarray()
                    NumNonZeroElementsByColumn = [X_clone2.sum(0)]
                    FPKM_count_lists += [list(np.array(NumNonZeroElementsByColumn)[0])]

            # Generate the count matrix df
            rna_count_lists = pd.DataFrame(rna_count_lists)
            rna_count_lists = rna_count_lists.transpose()
            rna_count_lists.columns = cluster_names
            rna_count_lists.index = adata.var_names
            rna_count_lists["average"] = rna_count_lists.mean(axis=1)
            rna_count_lists = rna_count_lists.astype("int")

            # Generate the FPKM matrix df
            FPKM_count_lists = pd.DataFrame(FPKM_count_lists)
            FPKM_count_lists = FPKM_count_lists.transpose()
            FPKM_count_lists.columns = cluster_names
            FPKM_count_lists.index = adata.var_names
            FPKM_count_lists["average"] = FPKM_count_lists.mean(axis=1)
            FPKM_count_lists = FPKM_count_lists.astype("int")

            count_file = str(outputdir +str(cond_name)+"_"+str(cluster_id)+"_RNA_Counts.tsv")
            CPM_file = str(outputdir +str(cond_name)+"_"+str(cluster_id)+ "_TPM.tsv")
            sample_file = str(outputdir +str(cond_name)+"_"+str(cluster_id)+ "_samples.tsv")
            rna_count_lists.to_csv(count_file, sep="\t", index=True, index_label=False)
            FPKM_count_lists.to_csv(CPM_file, sep="\t", index=True, index_label=False)
            sample_lists.to_csv(sample_file, sep="\t", index=False, index_label=False)
    
    # Import the intermediate results back
    for cluster_id in ["merged_batch","full_batch"]:
        logging.info('Running file merge with: '+str(cluster_id))
            
        # Merge the counts from the individual objects lists
        df_1 = pd.read_csv(str(outputdir +"cond1"+"_"+str(cluster_id)+"_RNA_Counts.tsv"),sep="\t")
        del df_1["average"]
        df_2 = pd.read_csv(str(outputdir +"cond2"+"_"+str(cluster_id)+"_RNA_Counts.tsv"),sep="\t")
        del df_2["average"]
        
        df_1_samples = pd.read_csv(str(outputdir +"cond1"+"_"+str(cluster_id)+ "_samples.tsv"),sep="\t")
        df_2_samples = pd.read_csv(str(outputdir +"cond2"+"_"+str(cluster_id)+ "_samples.tsv"),sep="\t")

        # Save merged count files
        merged_df = pd.merge(df_1, df_2, left_index=True, right_index=True, how='inner')
        merged_df.index.name='gene'
        
        logging.info('Saving merged count files for '+str(cluster_id))
        merged_file = str(outputdir+str(cluster_id)+"_merged.tsv")
        merged_df.to_csv(merged_file, sep="\t", index=True, index_label="gene")
        
        # Save merged sample files
        cond_samples = pd.concat([df_1_samples,df_2_samples])
        
        logging.info('Saving sample files for '+str(cluster_id))
        cond_samples_file = str(outputdir+str(cluster_id)+"_samples.tsv")
        cond_samples.to_csv(cond_samples_file, sep="\t", index=False)
        logging.info('Finished constructing contrast between conditions for: '+str(cluster_id))
    
    return

def predpars():
    # Create the parser
    parser = argparse.ArgumentParser(description='Run SVM prediction')

    # Add arguments
    parser.add_argument('--reference_H5AD', type=str, help='Path to reference H5AD file')
    parser.add_argument('--query_H5AD', type=str, help='Path to query H5AD file')
    parser.add_argument('--LabelsPath', type=str, help='Path to cell population annotations file')
    parser.add_argument('--OutputDir', type=str, help='Path to output directory')
    parser.add_argument('--rejected', dest='rejected', action='store_true', help='Use SVMrejected option')
    parser.add_argument('--Threshold_rej', type=float, default=0.7, help='Threshold used when rejecting cells, default is 0.7')
    parser.add_argument('--meta_atlas', dest='meta_atlas', action='store_true', help='Use meta_atlas data')

    # Parse the arguments
    args = parser.parse_args()
    
    # check that output directory exists, create it if necessary
    if not os.path.isdir(args.OutputDir):
        os.makedirs(args.OutputDir)

    # Call the svm_prediction function with the parsed arguments
    SVM_predict(
        args.reference_H5AD,
        args.query_H5AD,
        args.LabelsPath,
        args.OutputDir,
        args.rejected,
        args.Threshold_rej,
        args.meta_atlas)


def performpars():

    # Create the parser
    parser = argparse.ArgumentParser(description="Tests performance of SVM model based on a reference H5AD dataset.")
    parser.add_argument("--reference_H5AD", type=str, help="Path to the reference H5AD file")
    parser.add_argument("--LabelsPath", type=str, help="Path to the labels CSV file")
    parser.add_argument("--OutputDir", type=str, help="Output directory path")
    parser.add_argument('--rejected', dest='rejected', action='store_true', help='Use SVMrejected option')
    parser.add_argument("--Threshold_rej", type=float, default=0.7, help="Threshold value (default: 0.7)")
    parser.add_argument("--fold_splits", type=int, default=5, help="Number of fold splits for cross-validation (default: 5)")

    # Parse the arguments
    args = parser.parse_args()
    
    # check that output directory exists, create it if necessary
    if not os.path.isdir(args.OutputDir):
        os.makedirs(args.OutputDir)

    SVM_performance(
        args.reference_H5AD,
        args.LabelsPath,
        args.OutputDir,
        args.rejected,
        args.Threshold_rej,
        args.fold_splits)


def importpars():

    parser = argparse.ArgumentParser(description="Imports predicted results back to H5AD file")
    parser.add_argument("--query_H5AD", type=str, help="Path to query H5AD file")
    parser.add_argument("--OutputDir", type=str, help="Output directory path")
    parser.add_argument("--SVM_type", type=str, help="Type of SVM prediction (SVM or SVMrej)")
    parser.add_argument("--replicates", type=str, help="Replicates")
    parser.add_argument("--colord", type=str, help=".tsv file with meta-atlas order and colors")
    parser.add_argument("--meta-atlas", dest="meta_atlas", action="store_true", help="Use meta-atlas data")
    parser.add_argument("--show_bar", dest="show_bar", action="store_true", help="Plot barplot with SVM labels over specified replicates")

    args = parser.parse_args()

    SVM_import(
        args.query_H5AD,
        args.OutputDir,
        args.SVM_type,
        args.replicates,
        args.colord,
        args.meta_atlas,
        args.show_bar)
    

def pseudopars():

    parser = argparse.ArgumentParser(description="Performs individual and joined pseudobulk on two H5AD objects")
    parser.add_argument("--condition_1", type=str, help="Path to meta-atlas or other H5AD file")
    parser.add_argument("--condition_1_batch", type=str, help="Technical, biological or other replicate column in condition_1.obs")
    parser.add_argument("--condition_2", type=str, help="Path to H5AD file with SVM predictions")
    parser.add_argument("--condition_2_batch", type=str, help="Technical, biological or other replicate column in condition_2.obs")
    parser.add_argument("--Labels_1", type=str, help="Label path for the meta-atlas (LabelsPath) or condition_1.obs column with names")
    parser.add_argument('--OutputDir', dest='pseudobulk_output/', action='store_true', help='Directory where pseudobulk results are outputted')
    parser.add_argument("--min_cells", type=float, default=50, help="Minimal amount of cells for each condition and replicate")
    parser.add_argument("--SVM_type", type=str, help="Type of SVM prediction (SVM or SVMrej)")
    
    args = parser.parse_args()

    # check that output directory exists, create it if necessary
    if not os.path.isdir(args.OutputDir):
        os.makedirs(args.OutputDir)

    SVM_pseudobulk(
        args.condition_1,
        args.condition_1_batch,
        args.condition_2,
        args.condition_2_batch,
        args.Labels_1,
        args.OutputDir,
        args.min_cells,
        args.SVM_type)


if __name__ == '__predpars__':
    predpars()

    
if __name__ == '__performpars__':
    performpars()


if __name__ == '__importpars__':
    importpars()


if __name__ == '__pseudopars__':
    pseudopars()

