import argparse
import gc
import os
import numpy as np
import pandas as pd
import pyarrow as pa
import scanpy as sc
from scanpy import read_h5ad
import time as tm
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
import logging
import pickle
import joblib
import json

class CpredictorClassifier():
    def __init__(self, Threshold_rej, rejected, OutputDir):
        self.scaler = MinMaxScaler()
        self.Classifier = LinearSVC(C = 0.01, dual = False, random_state = 42, class_weight = 'balanced', max_iter = 1000)
        self.threshold = Threshold_rej
        self.rejected = rejected
        self.output_dir = OutputDir
        self.expression_treshold = 162
        self.kf = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 42)

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
        joblib.dump(clf, "data/model_SVMrej.pkl")
        self.Classifier.get_params()
        with open('data/params_SVMrej.json', 'w', encoding='utf-8') as outfile:
            json.dump(self.Classifier.get_params(), outfile)
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

    def predict_only_svmrejection(self, threshold, output_dir, data_test):
        self.rejected = True
        self.threshold = threshold
        self.output_dir = output_dir
        clf = joblib.load("data/model_SVMrej.pkl")
        logging.info('Running SVMrejection')
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
        joblib.dump(self.Classifier, "data/model_SVM.pkl")
        self.Classifier.get_params()
        with open('data/params_SVM.json', 'w', encoding='utf-8') as outfile:
            json.dump(self.Classifier.get_params(), outfile)
        self.predictions = self.Classifier.predict(data_test)
        self.save_results(self.rejected)

    def predict_only_svm(self, output_dir, data_test):
        self.rejected = False
        self.output_dir = output_dir
        clf = joblib.load("data/model_SVM.pkl")
        logging.info('Running SVM')
        self.predictions = clf.predict(data_test)
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

def SVM_predict(query_H5AD, LabelsPath, OutputDir, reference_H5AD=None, rejected=False, Threshold_rej=0.7, meta_atlas=False):
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
    meta_atlas : If the flag is added the predictions will use meta_atlas data.
    This means that reference_H5AD and LabelsPath do not need to be specified.
    '''
    logging.basicConfig(level=logging.DEBUG, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S',
                        filename=f'{OutputDir}/cPredictor_predict.log', filemode='w')
    
    # Get an instance of the Cpredictor class
    cpredictor = CpredictorClassifier(Threshold_rej, rejected, OutputDir)

    SVM_type = "SVMrej" if rejected is True else "SVM"

    # Load in the test data
    logging.info('Reading in the test data')
    testing = read_h5ad(query_H5AD)

    # Checks if the test data contains a raw data slot and sets it as the count value
    try:
        testing = testing.raw.to_adata()
    except AttributeError:
        logging.warning('Query object does not contain raw data, using sparce matrix from adata.X')
        logging.warning('Please manually check if this sparce matrix contains actual raw counts')

    # Checks if there is an actual column of "features"
    try:
        logging.warning('Going into the feature tryexcept')
        gene_sel = testing.var.features.values
    except AttributeError:
        gene_sel = testing.var.index.values
        testing.var['features'] = testing.var.index
        logging.debug('Using the var index as names of var features')

    matrix_test = pd.DataFrame.sparse.from_spmatrix(testing.X, index=list(testing.obs.index.values), columns=list(gene_sel))

    # If there is a pregenerated model the pipeline will try to run this first
    if os.path.exists(f"data/model_{SVM_type}.pkl") and meta_atlas is True:
        logging.info('Using a predifined model for predictions as well as for subselected genes')
        
        with open ('data/mergedgenes', 'rb') as fp:
            col_one_list = pickle.load(fp)

        missing_cols = list(set(col_one_list) - set(matrix_test.columns.to_list()))
        length_missing = len(missing_cols)

        if missing_cols:
            logging.warning(f'Filling in missing values as 0 in test data for {length_missing} genes')
            logging.warning('Please check the validity of your query H5AD object')
            matrix_test = matrix_test.reindex(col_one_list, axis=1)

            new_col_values = np.full(len(matrix_test), 0)
            for col in missing_cols:
                matrix_test[col] = new_col_values

        #matrix_test = matrix_test[matrix_test.columns.intersection(col_one_list)]
        matrix_test = matrix_test[list(col_one_list)]
        data_test = matrix_test.to_numpy(dtype="float16")

        logging.info('Processing test data')
        data_test = cpredictor.preprocess_data_test(data_test)

        if rejected is True:
            cpredictor.predict_only_svmrejection(Threshold_rej, OutputDir, data_test)
            cpredictor.save_results(rejected)
        
        else:
            cpredictor.predict_only_svm(OutputDir, data_test)
            cpredictor.save_results(rejected)

        return
    
    logging.info('Reading in the reference and query H5AD objects')
    # Load in the cma.h5ad object or use a different reference
    if meta_atlas is False:
        training = read_h5ad(reference_H5AD) 
    if meta_atlas is True:
        meta_dir = 'data/cma_meta_atlas.h5ad'
        training = read_h5ad(meta_dir) 

    training = cpredictor.expression_cutoff(training, LabelsPath)

    logging.info('Generating training matrix from the H5AD object')
    
    # training data
    matrix_train = pd.DataFrame.sparse.from_spmatrix(training.X, index=list(training.obs.index.values), columns=list(training.var.features.values))
    
    logging.info('Unifying training and testing matrices for shared genes')
    
    # subselect the train matrix for values that are present in both
    df_all = training.var[["features"]].merge(testing.var[["features"]].drop_duplicates(), on=['features'], how='left', indicator=True)
    df_all['_merge'] == 'left_only'
    training1 = df_all[df_all['_merge'] == 'both']
    col_one_list = training1['features'].tolist()

    # Save the list present in both
    matrix_test = matrix_test[matrix_test.columns.intersection(col_one_list)]
    matrix_train = matrix_train[matrix_train.columns.intersection(col_one_list)]
    matrix_train = matrix_train[list(matrix_test.columns)]

    # Save the list for future use in pretrained contrainer
    with open('data/mergedgenes', 'wb') as fp:
        pickle.dump(list(matrix_test.columns), fp)
    
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


def SVM_import(query_H5AD, OutputDir, SVM_type, replicates, sub_rep=None, colord=None, meta_atlas=False, show_bar=False, show_median=False):
    '''
    Imports the output of the SVM_predictor and saves it to the query_H5AD.

    Parameters:
    query_H5AD: H5AD file of datasets of interest.
    OutputDir: Output directory defining the path of the exported SVM_predictions.
    SVM_type: Type of SVM prediction, SVM (default) or SVMrej.
    Replicates: A string value specifying a column in query_H5AD.obs.
    colord: A .tsv file with the order of the meta_atlas and corresponding colors.
    meta_atlas : If the flag is added the predictions will use meta_atlas data.
    show_bar: Shows bar plots depending on the SVM_type, split over replicates.
    sub_rep:  A string value specifying an instance within the selected column in query_H5AD.obs.
    show_median: Shows the median values of the distribution of the predictions.

    '''
    logging.basicConfig(level=logging.DEBUG, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S',
                        filename=f'{OutputDir}/cPredictor_import.log', filemode='w')
    logging.info('Reading query data')

    # Makes a figure dir in the output dir if it does not exist yet
    if not os.path.isdir(f"{OutputDir}/figures"):
        os.makedirs(f"{OutputDir}/figures")
    
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
        sc.set_figure_params(figsize=(8, 5))
        
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
        fig.savefig(f"{OutputDir}/figures/{SVM_key}_bar.pdf", bbox_inches='tight')
        plt.close(fig)
    else:
        None

    if SVM_key == "SVM_predicted":
        logging.info('Plotting label prediction certainty scores')
        sns.set_theme(style='ticks')
        plt.rcParams['figure.figsize'] = 4,4

        # This allows for subsetting the density plot to individual instances of the replicates column
        if sub_rep is not None:
            adata = adata[adata.obs[replicates] == str(sub_rep)] # Add functional test here later

        # Iterate over each category and plot the density
        subset_joined = []
        for category, color in category_colors.items():
            subset = adata.obs[adata.obs['SVM_predicted'] == category]

            if show_median is True:
                label_name = f"{category} (Median: {subset['SVMrej_predicted_prob'].median():.2f})"
                
            if show_median is False:
                label_name = f"{category}"
                
            #subset[label_name] = subset["SVMrej_predicted_prob"]
            subset["Cell state"] = label_name
            subset = subset[["Cell state", "SVMrej_predicted_prob"]]
            subset_joined.append(subset)
            #ax.set(xlim=(0, 1))
        subset_joined = pd.concat(subset_joined)
        subset_joined = subset_joined.reset_index(drop=True)
        sns.displot(data=subset_joined, y="SVMrej_predicted_prob", hue="Cell state", kind="kde", cut=0, 
                    palette=category_colors, cumulative=True, common_norm=False, common_grid=True)

        # Set labels and title
        plt.xlabel('Density')
        plt.ylabel('SVM Certainty Scores')
        plt.title('Stacked Density Plots of Prediction Certainty Scores by Cell State')
        plt.ylim([0, 1])

        # Saving the density plot
        plt.savefig(f"{OutputDir}/figures/Density_prediction_scores.pdf", bbox_inches='tight')

    else:
        None
        
    logging.info('Saving H5AD file')
    adata.write_h5ad(f"{OutputDir}/{SVM_key}.h5ad")
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
                        filename=f'{OutputDir}/cPredictor_performance.log', filemode='w')

    logging.info('Reading in the data')

    # Makes a figure dir in the output dir if it does not exist yet
    if not os.path.isdir(f"{OutputDir}/figures"):
        os.makedirs(f"{OutputDir}/figures")

    Data = read_h5ad(reference_H5AD)

    # Using the child class of the CpredictorClassifier to process the data
    cpredictorperf = CpredictorClassifierPerformance(Threshold_rej, rejected, OutputDir)
    
    Data = cpredictorperf.expression_cutoff(Data, LabelsPath)

    data_train = pd.DataFrame.sparse.from_spmatrix(Data.X, index=list(Data.obs.index.values), columns=list(Data.var.features.values))
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
    sns.set_theme(font_scale=0.8)
    cm = sns.clustermap(cnf_matrix.T, cmap="Blues", annot=True,fmt='.2%', row_cluster=False,col_cluster=False)
    cm.savefig(f"{OutputDir}/figures/{SVM_type}_cnf_matrix.png")

    # Save classification report
    report =classification_report(true, pred, output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    df_classification_report = df_classification_report.sort_values(by=['f1-score'], ascending=False)
    df_classification_report.to_csv(f"{OutputDir}report.tsv", index=True, sep="\t")

    with open(f"{OutputDir}/metrics.txt", "w") as text_file:
        text_file.write(str(F1score)+"\n")
        text_file.write(str(acc_score)+"\n")
        text_file.write(str(prec_score)+"\n")
        text_file.close()
    
    return F1score,acc_score,prec_score


def predpars():

    # Create the parser
    parser = argparse.ArgumentParser(description='Run SVM prediction')

    # Add arguments
    parser.add_argument('--query_H5AD', type=str, help='Path to query H5AD file')
    parser.add_argument('--LabelsPath', type=str, help='Path to cell population annotations file')
    parser.add_argument('--OutputDir', type=str, help='Path to output directory')
    parser.add_argument('--reference_H5AD', type=str, help='Path to reference H5AD file')
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
        args.query_H5AD,
        args.LabelsPath,
        args.OutputDir,
        args.reference_H5AD,
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
    parser.add_argument("--sub_rep", type=str, help="Replicates")
    parser.add_argument("--colord", type=str, help=".tsv file with meta-atlas order and colors")
    parser.add_argument("--meta_atlas", dest="meta_atlas", action="store_true", help="Use meta-atlas data")
    parser.add_argument("--show_bar", dest="show_bar", action="store_true", help="Plot barplot with SVM labels over specified replicates")
    parser.add_argument("--show_median", dest="show_median", action="store_true", help="Shows median of scores")

    args = parser.parse_args()

    SVM_import(
        args.query_H5AD,
        args.OutputDir,
        args.SVM_type,
        args.replicates,
        args.sub_rep,
        args.colord,
        args.meta_atlas,
        args.show_bar,
        args.show_median)
    

if __name__ == '__predpars__':
    predpars()

    
if __name__ == '__performpars__':
    performpars()


if __name__ == '__importpars__':
    importpars()

