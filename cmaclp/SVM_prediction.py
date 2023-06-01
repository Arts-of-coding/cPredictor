import argparse
import os
import numpy as np
import pandas as pd
import scanpy as sc
import time as tm
import seaborn as sns
import rpy2.robjects as robjects
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
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
    if meta_atlas==False:
        training=read_h5ad(reference_H5AD) 
    if meta_atlas==True:
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
    if meta_atlas==True:
        LabelsPathTrain = files('cmaclp.data').joinpath('training_labels_meta.csv')
    
    labels_train = pd.read_csv(LabelsPathTrain, header=0,index_col=None, sep=',')
        
    # Set threshold for rejecting cells
    if rejected == True:
        Threshold = Threshold_rej

    print("Log normalizing the training and testing data")
    
    # normalise data
    data_train = np.log1p(data_train)
    data_test = np.log1p(data_test)  
        
    Classifier = LinearSVC()
    pred = []
    
    if rejected == True:
        print("Running SVMrejection")
        clf = CalibratedClassifierCV(Classifier, cv=3)
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
    
    if rejected == False:
        print("Running SVM")
        Classifier.fit(data_train, labels_train.values.ravel())
        predicted = Classifier.predict(data_test)    
        pred.extend(predicted)
        pred = pd.DataFrame(pred)
        
        # Save the predicted labels
        pred.to_csv(str(OutputDir) + "SVM_Pred_Labels.csv", index =False)

def SVM_prediction_import(query_H5AD, OutputDir, SVM_type, replicates, meta_atlas=True, show_umap=True, show_bar=True):
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
            if not 'rej' in file:
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

    # Plot UMAP if selected
    print("Plotting UMAP")
    sc.set_figure_params(figsize=(5, 5))
    if show_umap == True:
        if meta_atlas == True:
            category_order_list = ["LSC", "LESC","LE","Cj","CE","qSK","SK","TSK","CF","EC","Ves","Mel","IC","nm-cSC","MC","Unknown"]
            adata.obs[SVM_key] = adata.obs[SVM_key].astype("category")
            adata.obs[SVM_key] = adata.obs[SVM_key].cat.set_categories(category_order_list, ordered=True)
            sc.pl.umap(adata, color=SVM_key,palette={
                        "LSC": "#66CD00",
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
                        "Unknown": "#808080",
                        },show=False,save=f"_{SVM_key}.pdf")
        else:
            sc.pl.umap(adata, color=SVM_key,show=False,save=f"_{SVM_key}.pdf")
            
    # Plot absolute and relative barcharts across replicates
    print("Plotting barcharts")
    if show_bar == True:
        sc.set_figure_params(figsize=(15, 5))
        key=SVM_key
        obs_1 = key
        obs_2 = replicates

        n_categories = {x : len(adata.obs[x].cat.categories) for x in [obs_1, obs_2]}
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

        if meta_atlas == True:
            ordered_list=["LSC", "LESC","LE","Cj","CE","qSK","SK","TSK","CF","EC","Ves","Mel","IC","nm-cSC","MC","Unknown"]
            palette={
                        "LSC": "#66CD00",
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
                        "Unknown": "#808080",
                        }
            lstval = [palette[key] for key in ordered_list]
            sorter=ordered_list
            df = df[sorter]

        stacked_data = df.apply(lambda x: x*100/sum(x), axis=1)
        stacked_data=stacked_data.iloc[:, ::-1]

        fig, ax =plt.subplots(1,2)
        if meta_atlas == True:
            df.plot(kind="bar", stacked=True, ax=ax[0], legend = False,color=lstval, rot=45, title='Absolute number of cells')
        else:
            df.plot(kind="bar", stacked=True, ax=ax[0], legend = False, rot=45, title='Absolute number of cells')

        fig.legend(loc=7,title="Cell state")

        if meta_atlas == True:
            stacked_data.plot(kind="bar", stacked=True, legend = False, ax=ax[1],color=lstval[::-1], rot=45, title='Percentage of cells')
        else:
            stacked_data.plot(kind="bar", stacked=True, legend = False, ax=ax[1], rot=45, title='Percentage of cells')

        fig.tight_layout()
        fig.subplots_adjust(right=0.9)
        fig.savefig(f"figures/{SVM_key}_bar.pdf", bbox_inches='tight')
        plt.close(fig)
    else:
        None
    print("Saving H5AD file")
    adata.write_h5ad(f"{SVM_key}.h5ad")
    return

def main():
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
    SVM_prediction(args.reference_H5AD, args.query_H5AD, args.LabelsPathTrain, args.OutputDir, args.rejected, args.Threshold_rej, args.meta_atlas)

if __name__ == '__main__':
    main()
