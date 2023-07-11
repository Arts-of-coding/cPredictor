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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from scanpy import read_h5ad
from importlib.resources import files
import re
from statistics import mean
from scipy.stats import pearsonr
from scipy.stats import spearmanr

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
    print("Reading in the reference and query H5AD objects")
    
    # Load in the cma.h5ad object or use a different reference
    if meta_atlas is False:
        training=read_h5ad(reference_H5AD) 
    if meta_atlas is True:
        meta_dir='data/cma_meta_atlas.h5ad'
        training=read_h5ad(meta_dir) 

    # Load in the test data
    testing=read_h5ad(query_H5AD)

    # Checks if the test data contains a raw data slot and sets it as the count value
    try:
        testing=testing.raw.to_adata()
    except AttributeError:
        print("Query object does not contain raw data, using sparce matrix from adata.X")
        print("Please manually check if this sparce matrix contains actual raw counts")

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
        LabelsPath = 'data/training_labels_meta.csv'
    
    labels_train = pd.read_csv(LabelsPath, header=0,index_col=None, sep=',')
        
    # Set threshold for rejecting cells
    if rejected is True:
        Threshold = Threshold_rej

    print("Log normalizing the training and testing data")
    
    # normalise data
    data_train = np.log1p(data_train)
    data_test = np.log1p(data_test)  

    print("Scaling the training and testing data")
    scaler = MinMaxScaler()
    data_train = scaler.fit_transform(data_train)
    data_test = scaler.fit_transform(data_test)
    
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
        predicted[unlabeled] = 'Unlabeled'
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
    print("Reading query data")
    adata=read_h5ad(query_H5AD)
    SVM_key=f"{SVM_type}_predicted"

    # Load in the object and add the predicted labels
    print("Adding predictions to query data")
    for file in os.listdir(OutputDir):
        if file.endswith('.csv'):
            if 'rej' not in file:
                filedir= OutputDir+file
                SVM_output_dir= pd.read_csv(filedir,sep=',',index_col=0)
                SVM_output_dir=SVM_output_dir.index.tolist()
                adata.obs["SVM_predicted"]=SVM_output_dir
            if 'rej_Pred' in file:
                filedir= OutputDir+file
                SVM_output_dir= pd.read_csv(filedir,sep=',',index_col=0)
                SVM_output_dir=SVM_output_dir.index.tolist()
                adata.obs["SVMrej_predicted"]=SVM_output_dir
            if 'rej_Prob' in file:
                filedir= OutputDir+file
                SVM_output_dir= pd.read_csv(filedir,sep=',',index_col=0)
                SVM_output_dir=SVM_output_dir.index.tolist()
                adata.obs["SVMrej_predicted_prob"]=SVM_output_dir

    # Set category colors:
    if meta_atlas is True and colord is not None:
        df_category_colors=pd.read_csv(colord, header=None,index_col=False, sep='\t')
        category_colors = dict(zip(df_category_colors.iloc[:,0], df_category_colors.iloc[:,1]))
        if SVM_key == "SVMrej_predicted":
          category_colors["Unlabeled"]= "#808080"
                    
    if meta_atlas is False or colord is None:
    
      # Load a large color palette
      palette_name = "tab20"
      cmap = plt.get_cmap(palette_name)
      palette=[matplotlib.colors.rgb2hex(c) for c in cmap.colors] 
      
      # Extract the list of colors
      colors = palette
      key_cats = adata.obs[SVM_key].astype("category")
      key_list = key_cats.cat.categories.to_list()
            
      category_colors = dict(zip(key_list, colors[:len(key_list)]))
            
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

        if meta_atlas is True and colord is not None:
            palette=category_colors
            if SVM_type == 'SVM' :
              ord_list = [key for key in palette]
              
            if SVM_type == 'SVMrej':
              ord_list = [key for key in palette]
            
            # Sorts the df on the longer ordered list
            def sort_small_list(long_list, small_list):
              sorted_list = sorted(small_list, key=lambda x: long_list.index(x))
              return sorted_list
            
            sorter = sort_small_list(ord_list, df.columns.tolist())
            
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

    print("Scaling the data")
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
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
    print("Generate indices for train and test")
    for train_index, test_index in kfold.split(X):
        y_train, y_test = y[train_index], y[test_index]

        # Store the indices of the training and test sets for each fold
        train_indices.append(list(train_index))
        test_indices.append(list(test_index))

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
    print("Reading in the reference and query H5AD objects and adding batches")
    cond_1=read_h5ad(condition_1)
    cond_2=read_h5ad(condition_2)
    outputdir=OutputDir
    
    # Reading in the Labels_1 for the replicates
    # First tries to read in Labels_1 as a path
    try:
      label_data=pd.read_csv(Labels_1,sep=',',index_col=0)
      label_data=label_data.index.tolist()
      cond_1.obs["meta_atlas"]=label_data
      cond_1_label="meta_atlas"

    # Then tries to read in Labels_1 as a string in obs
    except (TypeError, FileNotFoundError) as error:
      try:
        cond_1_label=str(Labels_1)
        cond_1.obs[cond_1_label]

    # If no key if found a valid key must be entered
      except KeyError:
        raise ValueError('Please provide a valid name for labels in Labels_1')
      
    print("Constructing condition for condition 1")
    cond_1.obs["condition"]="cond1"

    print("Constructing batches for condition 1")

    # Add extra index to the ref object:
    cond_1.obs["batch"] = cond_1.obs[condition_1_batch]

    # Convert the 'Category' column to numeric labels
    cond_1.obs["batch"] = pd.factorize(cond_1.obs["batch"])[0] + 1

    cond_1.obs["merged"]=cond_1.obs[cond_1_label]+"."+cond_1.obs["condition"]
    cond_1.obs["merged_batch"]=cond_1.obs["merged"]+"."+cond_1.obs["batch"].astype(str)
    cond_1.obs["full_batch"]=cond_1.obs["condition"]+"."+cond_1.obs["batch"].astype(str)

    print("Constructing batches for condition 2")

    # Uses the specified SVM_type for predicted cond_2
    cond_2_label=f'{SVM_type}_predicted'
    batch=condition_2_batch

    cond_2.obs["condition"]="cond2"
    cond_2.obs["batch"]=cond_2.obs[batch]
    cond_2.obs["merged"]=cond_2.obs[cond_2_label].astype(str)+"."+cond_2.obs["condition"]
    cond_2.obs["merged_batch"]=cond_2.obs["merged"]+"."+cond_2.obs["batch"].astype(str)
    cond_2.obs["full_batch"]=cond_2.obs["condition"]+"."+cond_2.obs["batch"].astype(str)
    
    os.makedirs(outputdir, exist_ok=True)
        
    # Use each object as a condition
    for cond in cond_1,cond_2:
        cond_str = cond.obs["condition"].astype(str)[1]
        print(f"Running with {cond_str}")

        if cond.obs["condition"].astype(str)[1] == "cond1":
            cond_name="cond1"
        elif cond.obs["condition"].astype(str)[1] == "cond2":
            cond_name="cond2"

        # Construct the sample.tsv file    
        cond.obs["assembly"]="hg38"

        adata = cond.copy()

        # Extract names of genes
        try:
            adata.var_names=adata.var["_index"].tolist()
        except KeyError:
            adata.var_names=adata.var.index


        for cluster_id in ["merged_batch","full_batch"]:
            print("Running pseudobulk extraction with: "+str(cluster_id))
            if cluster_id == "full_batch":
                sample_lists=adata.obs[[cluster_id,"assembly","condition","full_batch"]]
            if cluster_id == "merged_batch":
                sample_lists=adata.obs[[cluster_id,"assembly","merged","condition","full_batch"]]  
            sample_lists=sample_lists.reset_index(drop=True)
            sample_lists=sample_lists.drop_duplicates()
            sample_lists.rename(columns={ sample_lists.columns[0]: "sample" }, inplace = True)

            rna_count_lists = list()
            FPKM_count_lists = list()
            cluster_names = list()

            for cluster in adata.obs[cluster_id].astype("category").unique():

                # Only use ANANSE on clusters with more than minimal amount of cells
                n_cells = adata.obs[cluster_id].value_counts()[cluster]

                if n_cells > min_cells:
                    cluster_names.append(str(cluster))

                    # Generate the raw count file
                    adata_sel = adata[adata.obs[cluster_id].isin([cluster])].copy()
                    adata_sel.raw=adata_sel

                    print(
                        str("gather data from " + cluster + " with " + str(n_cells) + " cells")
                    )

                    X_clone = adata_sel.X.tocsc()
                    X_clone.data = np.ones(X_clone.data.shape)
                    NumNonZeroElementsByColumn = X_clone.sum(0)
                    rna_count_lists += [list(np.array(NumNonZeroElementsByColumn)[0])]
                    sc.pp.normalize_total(adata_sel, target_sum=1e6, inplace=True)
                    X_clone2=adata_sel.X.toarray()
                    NumNonZeroElementsByColumn = [X_clone2.sum(0)]
                    FPKM_count_lists += [list(np.array(NumNonZeroElementsByColumn)[0])]

            # Specify the df.index
            df = adata.T.to_df()

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
        print("Running file merge with: "+str(cluster_id))
            
        # Merge the counts from the individual objects lists
        df_1 = pd.read_csv(str(outputdir +"cond1"+"_"+str(cluster_id)+"_RNA_Counts.tsv"),sep="\t")
        del df_1["average"]
        df_2 = pd.read_csv(str(outputdir +"cond2"+"_"+str(cluster_id)+"_RNA_Counts.tsv"),sep="\t")
        del df_2["average"]
        
        df_1_samples=pd.read_csv(str(outputdir +"cond1"+"_"+str(cluster_id)+ "_samples.tsv"),sep="\t")
        df_2_samples=pd.read_csv(str(outputdir +"cond2"+"_"+str(cluster_id)+ "_samples.tsv"),sep="\t")

        # Save merged count files
        merged_df = pd.merge(df_1, df_2, left_index=True, right_index=True, how='inner')
        merged_df.index.name='gene'
        
        print("Saving merged count files for "+str(cluster_id))
        merged_file= str(outputdir+str(cluster_id)+"_merged.tsv")
        merged_df.to_csv(merged_file, sep="\t", index=True, index_label="gene")
        
        # Save merged sample files
        cond_samples = pd.concat([df_1_samples,df_2_samples])
        
        print("Saving sample files for "+str(cluster_id))
        cond_samples_file= str(outputdir+str(cluster_id)+"_samples.tsv")
        cond_samples.to_csv(cond_samples_file, sep="\t", index=False)
        print("Finished constructing contrast between conditions for: "+str(cluster_id))
    
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

