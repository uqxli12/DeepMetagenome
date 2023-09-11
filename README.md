# DeepMetProteome
A deep learning package for mining specific proteins from metaproteomes/metagenomes
# how to use: preparation
1. download the package
2. unzip the package to a desired work directory
3. preparation of the platform: Anaconda, tensorflow 2.0, Python 3.7, and all dependencies for deep learning
4. preparation of the binary training dataset
# the package
the package contains all modules for parsing, merging proteome datasets, sequence statistics, building and training deep learning models, prediction and processing of output data
# how to use: scripts
## in Jupyter Notebook, go to the DeepMetagenome directory
!cd ../DeepMetagenome
## there are three variables subjected to configuration, MinLen or MaxLen, CutOffValue. New variables can also be added and modified the modules. For MT, we use MaxLen = 200 and CutOffValue = 0.000001. The length selection is a necessary step in preparing traning dataset, and this cutoff value is a result of the sequence statistics. The probability value is an arbitrary value, depending on the sequence features and purpose of subsequent study.
## call all the functions 
$from Main import *
## create a folder saving all the prediction dataset. Training dataset should be in the current work directory, or elsewhere and use absolute address
$preprocessing_for_multi_FASTA("../pred_dataset")
$training_dataset_preprocess("MTtraningbinaryDatabase.csv")
## start to build model and prediction
$my_model()
$model_pred()
## an output file can be seen in the work directory. The file is a .csv file with three features of ‘index’, ‘probability’ and ‘sequence’.
# Contacts
Xiaofang Li (xfli@sjziam.ac.cn; xiaofang_lee@163.com)
