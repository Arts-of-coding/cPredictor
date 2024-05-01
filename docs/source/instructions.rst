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
you can use the first version of the cornea meta-atlas (hcornea_v1) together with the latest version of cPredictor (e.g. v0.3.5).
The docker containers are organised in this fashion: {version}_{tool}_{dataset}_{version}, like v0.3.5_cpredictor_hcornea_v1.

From the command-prompt you can pull the container.

.. code-block:: console

   $ docker pull artsofcoding/cpredictor:v0.3.2_cpredictor_hcornea_v1

Then you can navigate to the directory (current directory (%cd%) on Windows) where you have your dataset of interest in the root.
This command makes the data load correctly into the test folder.

.. code-block:: console

   $ docker run -it -v %cd%:/test --entrypoint /bin/sh artsofcoding/cpredictor:v0.3.2_cpredictor_hcornea_v1
   
Then you can predict labels in your dataset of interest using the non-rejected version (no certainty score) or with "--rejected". 

.. code-block:: console

   $ SVM_predict --query_H5AD /test/{test_data}.h5ad --OutputDir cPredictor_output/ --meta_atlas
   $ SVM_predict --query_H5AD /test/{test_data}.h5ad --OutputDir cPredictor_output/ --meta_atlas --rejected

After the predictions you can import them back into your object. With "--replicates" and "--show_bar" you automatically plot biological or 
technical replicate columns (adata.obs[column]) together with the predictions.
This can for instance be "batch" or "study", but also "time_point" like in the example below.

.. code-block:: console

   $ SVM_import --query_H5AD /test/small_test.h5ad --OutputDir cPredictor_output/ --colord data/colord.tsv --SVM_type SVM --replicates time_point --meta_atlas --show_bar
   $ SVM_import --query_H5AD /test/small_test.h5ad --OutputDir cPredictor_output/ --colord data/colord.tsv --SVM_type SVMrej --replicates time_point --meta-atlas --show_bar
   
.. _usage:

Download atlases
------------
Documentation will be extended on how to download meta-atlases to load into cPredictor.

Please use the most recent version. Previous versions are included for completeness.