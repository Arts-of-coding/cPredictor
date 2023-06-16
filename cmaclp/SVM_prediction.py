import argparse
import os
import numpy as np
import pandas as pd
import scanpy as sc
import time as tm
import seaborn as sns
import rpy2.robjects as robjects
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

def SVM_prediction(reference_H5AD, query_H5AD, LabelsPathTrain, OutputDir, rejected=False, Threshold_rej=0.7,meta_atlas=False):
    '''
    run baseline classifier: SVM
    Wrapper script to run an SVM classifier with a linear kernel on a benchmark dataset with 5-fold cross validation,
    outputs lists of true and predicted cell labels as csv files, as well as computation time.

    Parameters:
    reference_H5AD, query_H5AD : H5AD files that produce training and testing data,
        cells-genes matrix with cell unique barcodes as row names and gene names as column names.
    LabelsPathTrain : Cell population annotations file path matching the training data (.csv).
    OutputDir : Output directory defining the path of the exported file.
    rejected: If the flag is added, then the SVMrejected option is chosen. Default: False.
    Threshold_rej : Threshold used when rejecting the cells, default is 0.7.
    meta_atlas : If the flag is added the predictions will use the corneal meta-atlas data,
    meaning that reference_H5AD and LabelsPathTrain do not need to be specified.
    '''
    print("Reading in the reference and query H5AD objects")
    
    # Load in the cma.h5ad object or use a different reference
    if meta_atlas is False:
        training=read_h5ad(reference_H5AD) 
    if meta_atlas is True:
    #    if not if os.path.exists(DEG_file):
    #        if not outputdir == "":
        os.makedirs(OutputDir, exist_ok=True)
        meta_dir=files('cmaclp.data').joinpath('cma_meta_atlas.h5ad')
        training=read_h5ad(meta_dir) 

    # Load in the test data
    testing=read_h5ad(query_H5AD)
 
    print("Generating training and testing matrices from the H5AD objects")
    
    # training data
    matrix_train = pd.DataFrame.sparse.from_spmatrix(training.X, index=list(training.obs.index.values), columns=list(training.var.features.values))

    # testing data
    try: 
        testing.var['features']
    except KeyError:
        testing.var['features'] = testing.var.index
    
    matrix_test = pd.DataFrame.sparse.from_spmatrix(testing.X, index=list(testing.obs.index.values), columns=list(testing.var.features.values))
    
    print("Unifying training and testing matrices for shared genes")
    
    # subselect the train matrix for values that are present in both
    df_all = training.var[["features"]].merge(testing.var[["features"]].drop_duplicates(), on=['features'], how='left', indicator=True)
    df_all['_merge'] == 'left_only'
    training1 = df_all[df_all['_merge'] == 'both']
    col_one_list = training1['features'].tolist()

    matrix_test = matrix_test[matrix_test.columns.intersection(col_one_list)]
    matrix_train = matrix_train[matrix_train.columns.intersection(col_one_list)]
    matrix_train = matrix_train[list(matrix_test.columns)]
    
    print("Number of genes remaining after unifying training and testing matrices: "+str(len(matrix_test.columns)))
    
    # Convert the ordered dataframes back to nparrays
    matrix_train2 = matrix_train.to_numpy()
    matrix_test2 = matrix_test.to_numpy()
    
    # Delete large objects from memory
    del matrix_train, matrix_test, training, testing
    
    # read the data
    data_train = matrix_train2
    data_test = matrix_test2
    
    # If meta_atlas=True it will read the training_labels
    if meta_atlas is True:
        LabelsPathTrain = files('cmaclp.data').joinpath('training_labels_meta.csv')
    
    labels_train = pd.read_csv(LabelsPathTrain, header=0,index_col=None, sep=',')
        
    # Set threshold for rejecting cells
    if rejected is True:
        Threshold = Threshold_rej

    print("Log normalizing the training and testing data")
    
    # normalise data
    data_train = np.log1p(data_train)
    data_test = np.log1p(data_test)  
        
    Classifier = LinearSVC()
    pred = []
    
    if rejected is True:
        print("Running SVMrejection")
        kf = KFold(n_splits=3)
        clf = CalibratedClassifierCV(Classifier, cv=kf)
        probability = [] 
        clf.fit(data_train, labels_train.values.ravel())
        predicted = clf.predict(data_test)
        prob = np.max(clf.predict_proba(data_test), axis = 1)
        unlabeled = np.where(prob < Threshold)
        predicted[unlabeled] = 'Unknown'
        pred.extend(predicted)
        probability.extend(prob)
        pred = pd.DataFrame(pred)
        probability = pd.DataFrame(probability)
        
        # Save the labels and probability
        pred.to_csv(str(OutputDir) + "SVMrej_Pred_Labels.csv", index = False)
        probability.to_csv(str(OutputDir) + "SVMrej_Prob.csv", index = False)
    
    if rejected is False:
        print("Running SVM")
        Classifier.fit(data_train, labels_train.values.ravel())
        predicted = Classifier.predict(data_test)    
        pred.extend(predicted)
        pred = pd.DataFrame(pred)
        
        # Save the predicted labels
        pred.to_csv(str(OutputDir) + "SVM_Pred_Labels.csv", index =False)


def SVM_import(query_H5AD, OutputDir, SVM_type, replicates, meta_atlas=False, show_umap=False, show_bar=False):
    '''
    Imports the output of the SVM_predictor and saves it to the query_H5AD.

    Parameters:
    query_H5AD : H5AD file of datasets of interest.
    OutputDir : Output directory defining the path of the exported SVM_predictions.
    SVM_type: Type of SVM prediction, SVM (default) or SVMrej.
    Replicates: 
    meta_atlas:
    show_umap:
    show_bar:
    '''
    print("Reading query data")
    adata=read_h5ad(query_H5AD)
    SVM_key=f"{SVM_type}_predicted"

    # Load in the object and add the predicted labels
    print("Adding predictions to query data")
    for file in os.listdir(OutputDir):
        if file.endswith('.csv'):
            if 'rej' not in file:
                filedir= OutputDir+file
                #print(filedir)
                influence_data= pd.read_csv(filedir,sep=',',index_col=0)
                #print(influence_data)
                influence_data=influence_data.index.tolist()
                adata.obs["SVM_predicted"]=influence_data
            if 'rej_Pred' in file:
                filedir= OutputDir+file
                #print(filedir)
                influence_data= pd.read_csv(filedir,sep=',',index_col=0)
                #print(influence_data)
                influence_data=influence_data.index.tolist()
                adata.obs["SVMrej_predicted"]=influence_data
            if 'rej_Prob' in file:
                filedir= OutputDir+file
                #print(filedir)
                influence_data= pd.read_csv(filedir,sep=',',index_col=0)
                #print(influence_data)
                influence_data=influence_data.index.tolist()
                adata.obs["SVMrej_predicted_prob"]=influence_data

    # Set category colors:
    if meta_atlas is True:
        category_colors = {"LSC": "#66CD00",
                    "LESC": "#76EE00",
                    "LE": "#66CDAA",
                    "Cj": "#191970",
                    "CE": "#1874CD",
                    "qSK": "#FFB90F",
                    "SK": "#EEAD0E",
                    "TSK": "#FF7F00",
                    "CF": "#CD6600",
                    "EC": "#87CEFA",
                    "Ves": "#8B2323",
                    "Mel": "#FFFF00",
                    "IC": "#00CED1",
                    "nm-cSC": "#FF0000",
                    "MC": "#CD3700",
                    "Unknown": "#808080"}
                    
    if meta_atlas is False:
    
      # Load a large color palette
      palette_name = "tab20"
      cmap = plt.get_cmap(palette_name)
      palette=[matplotlib.colors.rgb2hex(c) for c in cmap.colors] 
      
      # Extract the list of colors
      colors = palette
      key_cats = adata.obs[SVM_key].astype("category")
      key_list = key_cats.cat.categories.to_list()
            
      category_colors = dict(zip(key_list, colors[:len(key_list)]))
      
    # Plot UMAP if selected
    print("Plotting UMAP")
    sc.set_figure_params(figsize=(5, 5))
    if show_umap is True:
        if meta_atlas is True:
            category_order_list = ["LSC", "LESC","LE","Cj","CE","qSK","SK","TSK","CF","EC","Ves","Mel","IC","nm-cSC","MC","Unknown"]
            adata.obs[SVM_key] = adata.obs[SVM_key].astype("category")
            adata.obs[SVM_key] = adata.obs[SVM_key].cat.set_categories(category_order_list, ordered=True)
            sc.pl.umap(adata, color=SVM_key,palette=category_colors,show=False,save=f"_{SVM_key}.pdf")
        else:
            sc.pl.umap(adata, color=SVM_key,show=False,save=f"_{SVM_key}.pdf")
            
    # Plot absolute and relative barcharts across replicates
    print("Plotting barcharts")
    if show_bar is True:
        sc.set_figure_params(figsize=(15, 5))
        
        key=SVM_key
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

        for b, l in df:
            obs2_to_obs1[b][obs1_clusters.index(str(l))] += 1
            obs1_to_obs2[l][obs2_clusters.index(str(b))] += 1

        df = pd.DataFrame.from_dict(obs2_to_obs1,orient = 'index').reset_index()
        df = df.set_index(["index"])
        df.columns=obs1_clusters
        df.index.names = ['Replicate']

        if meta_atlas is True:
            palette=category_colors
            if SVM_type == 'SVM':
              ordered_list=["LSC", "LESC","LE","Cj","CE","qSK","SK","TSK","CF","EC","Ves","Mel","IC","nm-cSC","MC"]
            if SVM_type == 'SVMrej':
              ordered_list=["LSC", "LESC","LE","Cj","CE","qSK","SK","TSK","CF","EC","Ves","Mel","IC","nm-cSC","MC","Unknown"]
            
            # Sorts the df on the longer ordered list
            sorter=sorted(df.columns, key=ordered_list.index)
            
            # Retrieve the color codes from the sorted list
            lstval = [palette[key] for key in sorter]
            
            try:
              df=df[sorter]
            except KeyError:
              df=df
        else:
            lstval=list(category_colors.values())

        stacked_data = df.apply(lambda x: x*100/sum(x), axis=1)
        stacked_data=stacked_data.iloc[:, ::-1]

        fig, ax =plt.subplots(1,2)
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
        print("Plotting label prediction certainty scores")
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
        #fig.legend(loc=7,title="Cell states and median predictions scores")
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

        # Saving the density plot
        fig.savefig("figures/Density_prediction_scores.pdf", bbox_inches='tight')
        plt.close(fig)
    else:
        None
        
    print("Saving H5AD file")
    adata.write_h5ad(f"{SVM_key}.h5ad")
    return


def SVM_performance(reference_H5AD, OutputDir, LabelsPath, SVM_type="SVMrej", fold_splits=5, Threshold=0.7):
    '''
    Tests performance of SVM model based on a reference H5AD dataset.

    Parameters:
    reference_H5AD : H5AD file of datasets of interest.
    OutputDir : Output directory defining the path of the exported SVM_predictions.
    SVM_type: Type of SVM prediction, SVM or SVMrej (default).
    '''

    print("Reading in the data")
    Data=read_h5ad(reference_H5AD)

    data = pd.DataFrame.sparse.from_spmatrix(Data.X, index=list(Data.obs.index.values), columns=list(Data.var.index.values))

    labels = pd.read_csv(LabelsPath, header=0,index_col=None, sep=',') #, usecols = col

    # Convert the ordered dataframes back to nparrays
    print("Normalising the data")
    data = data.to_numpy(dtype="float16")
    np.log1p(data,out=data)

    X = data
    del data

    label_encoder = LabelEncoder()
    
    y = label_encoder.fit_transform(labels["x"].tolist())

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
    print("Generate indices for train and test")
    for train_index, test_index in kfold.split(X):
        #X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Store the indices of the training and test sets for each fold
        train_indices.append(list(train_index))
        test_indices.append(list(test_index))
        
    #train_indices = list(train_indices)
    #test_indices = list(test_indices)

    # Run the SVM model
    test_ind=test_indices
    train_ind=train_indices
    Classifier = LinearSVC()

    if SVM_type == "SVMrej":
        clf = CalibratedClassifierCV(Classifier, cv=3)

    tr_time=[]
    ts_time=[]
    truelab = []
    pred = []
    prob_full = []

    for i in range(fold_splits):
        print(f"Running cross-val {i}")
        train=X[train_ind[i]] # was data
        test=X[test_ind[i]] # was data
        y_train=y[train_ind[i]]
        y_test=y[test_ind[i]]

        if SVM_type == "SVMrej":
            start=tm.time()
            clf.fit(train, y_train.ravel()) #.values
            tr_time.append(tm.time()-start)

            start=tm.time()
            predicted = clf.predict(test)
            prob = np.max(clf.predict_proba(test), axis = 1)

            unlabeled = np.where(prob < float(Threshold))
            unlabeled=list(unlabeled[0])
            predicted[unlabeled] = 999999 # set arbitrary value to convert it back to a string in the end
            ts_time.append(tm.time()-start)
            pred.extend(predicted)
            prob_full.extend(prob)

        if SVM_type == "SVM":
            start=tm.time()
            Classifier.fit(train, y_train.ravel())
            tr_time.append(tm.time()-start)

            start=tm.time()
            predicted = Classifier.predict(test)
            ts_time.append(tm.time()-start)

            truelab.extend(y_test.values)
            pred.extend(predicted)

        truelab.extend(y_test)

    truelab = pd.DataFrame(truelab)
    pred = pd.DataFrame(pred)

    tr_time = pd.DataFrame(tr_time)
    ts_time = pd.DataFrame(ts_time)

    # Calculating the weighted F1 score:
    F1score= f1_score(truelab[0].to_list(), pred[0].to_list(), average='weighted')
    print(f"The {SVM_type} model ran with the weighted F1 score of: {F1score}")

    # Calculating the weighted accuracy score:
    acc_score = accuracy_score(truelab[0].to_list(), pred[0].to_list())
    print(f"The {SVM_type} model ran with the weighted accuracy score of: {acc_score}")

    # Calculating the weighted precision score:
    prec_score = precision_score(truelab[0].to_list(),  pred[0].to_list(),average="weighted")
    print(f"The {SVM_type} model ran with the weighted precision score of: {prec_score}")

    # Relabel truelab and predicted values by names
    truelab[0]=truelab[0].replace(res)
    pred[0]=pred[0].replace(res)

    print("Saving labels to specified output directory")
    truelab.to_csv(f"{OutputDir}{SVM_type}_True_Labels.csv", index = False)
    pred.to_csv(f"{OutputDir}{SVM_type}_Pred_Labels.csv", index = False)
    tr_time.to_csv(f"{OutputDir}{SVM_type}_Training_Time.csv", index = False)
    ts_time.to_csv(f"{OutputDir}{SVM_type}_Testing_Time.csv", index = False)

    ## Plot the SVM figures
    print("Plotting the confusion matrix")
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


def predpars():
    # Create the parser
    parser = argparse.ArgumentParser(description='Run SVM prediction')

    # Add arguments
    parser.add_argument('--reference_H5AD', type=str, help='Path to reference H5AD file')
    parser.add_argument('--query_H5AD', type=str, help='Path to query H5AD file')
    parser.add_argument('--LabelsPathTrain', type=str, help='Path to cell population annotations file')
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
    SVM_prediction(
        args.reference_H5AD,
        args.query_H5AD,
        args.LabelsPathTrain,
        args.OutputDir,
        args.rejected,
        args.Threshold_rej,
        args.meta_atlas)


def performpars():

    # Create the parser
    parser = argparse.ArgumentParser(description="Tests performance of SVM model based on a reference H5AD dataset.")
    parser.add_argument("--reference_H5AD", type=str, help="Path to the reference H5AD file")
    parser.add_argument("--OutputDir", type=str, help="Output directory path")
    parser.add_argument("--LabelsPath", type=str, help="Path to the labels CSV file")
    parser.add_argument("--SVM_type", default="SVMrej", help="Type of SVM prediction (default: SVMrej)")
    parser.add_argument("--fold_splits", type=int, default=5, help="Number of fold splits for cross-validation (default: 5)")
    parser.add_argument("--Threshold", type=float, default=0.7, help="Threshold value (default: 0.7)")
    args = parser.parse_args()

    SVM_performance(
        args.reference_H5AD,
        args.OutputDir,
        args.LabelsPath,
        SVM_type=args.SVM_type,
        fold_splits=args.fold_splits,
        Threshold=args.Threshold)


def importpars():

    parser = argparse.ArgumentParser(description="Imports predicted results back to H5AD file")
    parser.add_argument("--query_H5AD", type=str, help="Path to query H5AD file")
    parser.add_argument("--OutputDir", type=str, help="Output directory path")
    parser.add_argument("--SVM_type", type=str, help="Type of SVM prediction (SVM or SVMrej)")
    parser.add_argument("--replicates", type=str, help="Replicates")
    parser.add_argument("--meta-atlas", dest="meta_atlas", action="store_true", help="Use meta atlas data")
    parser.add_argument("--show-umap", dest="show_umap", action="store_true", help="Show UMAP plotting")
    parser.add_argument("--show-bar", dest="show_bar", action="store_true", help="Show bar chart plotting")

    args = parser.parse_args()

    SVM_import(
        args.query_H5AD,
        args.OutputDir,
        args.SVM_type,
        args.replicates,
        args.meta_atlas,
        args.show_umap,
        args.show_bar)


if __name__ == '__predpars__':
    predpars()

    
if __name__ == '__performpars__':
    performpars()


if __name__ == '__importpars__':
    importpars()

