Instructions
===== 

.. _instructions:
CLI commands
------------
cPredictor's cli commands are similar across distributions.

Two functions are needed to predict on your query dataset of interest and to import the predictions back into your object.
To see all options from these commands, please run:

.. code-block:: console

   $ SVM_predict --help
   $ SVM_import --help

Another function can be used to automatically retrieve accuracy, recall and F1 scores by using a 5-fold cross-validation on training data.
To see all options from this command, please run:

.. code-block:: console

   $ SVM_performance --help

There is also a fuction currently in development to directly make pseudobulk tables from predicted cell states compared to another single-cell object of interest.
Documentation of this function will be extended in future versions.

.. code-block:: console

   $ SVM_pseudobulk --help

Docker
------------
Pre-trained models can be run on your data. For instance for human corneal datasets, 
you can use the first version of the cornea meta-atlas (hcornea_v1) together with the latest version of cPredictor (e.g. v0.4.5).

The docker containers are organised in this fashion: {version}_{tool}_{dataset}_{version}, like v0.4.5_cpredictor_hcornea_v1.

For Docker to be able to run, you need to install it on your system and have wsl2 installed too. Docker can be run from both Docker Desktop or Rancher Desktop.

From the command-prompt you can pull the container.

.. code-block:: console

   $ docker pull artsofcoding/cpredictor:v0.4.5_cpredictor_hcornea_v1

Then you can navigate to the directory (current directory (%cd%) on Windows) where you have your dataset of interest in the root.
This command makes the data load correctly into the test folder.

.. code-block:: console

   $ docker run -it  -v ${PWD}:/app/test --entrypoint /bin/sh artsofcoding/cpredictor:v0.4.5_cpredictor_hcornea_v1
   
Then you can predict labels in your dataset of interest using the non-rejected version (no certainty score) or with "--rejected". 

.. code-block:: console

   $ SVM_predict --query_H5AD test/{test_data}.h5ad --OutputDir cPredictor_output/ --meta_atlas
   $ SVM_predict --query_H5AD test/{test_data}.h5ad --OutputDir cPredictor_output/ --meta_atlas --rejected

After the predictions you can import them back into your object. With "--replicates" and "--show_bar" you automatically plot biological or 
technical replicate columns (adata.obs[column]) together with the predictions.
This can for instance be "batch" or "study", but also "time_point" like in the example below.

.. code-block:: console

   $ SVM_import --query_H5AD test/{test_data}.h5ad --OutputDir cPredictor_output/ --colord data/colord.tsv --SVM_type SVM --replicates time_point --meta_atlas
   $ SVM_import --query_H5AD test/{test_data}.h5ad --OutputDir cPredictor_output/ --colord data/colord.tsv --SVM_type SVMrej --replicates time_point --meta-atlas

Docker & Azure
------------
Instead of running the application fully local, it is possible to couple Azure as a cloud service. The Docker container uses Blobstorage.

You first define a .env file with Azure storage details.

.. code-block:: console

   AZURE_STORAGE_ACCOUNT={name}
   AZURE_STORAGE_ACCESS_KEY={key}
   AZURE_STORAGE_ACCOUNT_CONTAINER={name_container}
   AZURE_MOUNT_POINT=/app/azure

Next, you can use this .env file to directly couple locally run cPredictor to Azure.

.. code-block:: console

   $ docker run -i -t --env-file .env --privileged artsofcoding/cpredictor:v0.4.5_cpredictor_hcornea_v1 --entrypoint /bin/bash

Then from the cPredictor container you can directly run it as an Azure directory. Note that this must be coupled to your AZURE_MOUNT_POINT configuration.

.. code-block:: console

   $ SVM_predict --query_H5AD azure/{test_data}.h5ad --OutputDir cPredictor_output/ --meta_atlas

Using other atlases
------------
cPredictor is able to work with other meta-atlases as well. This requires you to have several files in the data folder.

The first is your constructed meta-atlas.h5ad object. If you rename your object of interest to "cma_meta_atlas.h5ad", cPredictor will pick it up. Second you need a "training_labels_meta.csv" which specifies the cluster names of your meta-atlas of interest. Note that this needs to be 1 row longer than the number of cells, because of an expected header. Third, cPredictor provides the option to incorporate a predefined color scheme. If you do not provide this, then you cannot use the --colord flag in the import function. If you do want this, you can specify a colord.tsv file like this:

.. code-block:: console

   LSC-1	#66CD00
   LSC-2	#76EE00
   LE	#66CDAA
   Cj	#191970
   CE	#1874CD
   qSK	#FFB90F
   SK	#EEAD0E
   TSK	#FF7F00
   CF	#CD6600
   EC	#87CEFA
   Ves	#8B2323
   Mel	#FFFF00
   IC	#00CED1
   nm-cSC	#FF0000
   MC	#CD3700

Downloading atlases
------------
For now, the only atlas to download is the human cornea meta-atlas (v1). The files can be fully downloaded from Zenodo: https://doi.org/10.5281/zenodo.7970736. This is only needed for non-containerised use. Containers will have these files within /app/data.

With curl or wget you can download the files from these links:

.. code-block:: console

   $ wget https://zenodo.org/records/14536656/files/training_labels_meta.csv?download=1
   $ wget https://zenodo.org/records/14536656/files/cma_meta_atlas_rfe.h5ad?download=1 # This is for the cornea meta-atlas with genes after RFE.
   $ wget https://zenodo.org/records/14536656/files/colord.tsv?download=1 # Optional if you want identical colors to the manuscript.

.. _usage:




Please use the most recent version. Previous versions are included for completeness.
