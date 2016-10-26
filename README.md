#  EP2vec

EP2vec is computational framework to predict enhancer-promoter interactions by extracting “sequence embedding features”, defined as fixed-length vector representations learned from variable-length sequences, via a deep learning method in natural language processing named Paragraph Vector. (EP2vec: a deep learning approach for extracting sequence embedding features to predict enhancer-promoter interactions)


#  Traing Data

EP2vec uses the same training data as TargetFinder, where interacting enhancer-promoter pairs are annotated using high-resolution genome-wide Hi-C data (Rao et al., 2014). Labeled training datasets used in the TargetFinder are available in https://github.com/shwhalen/targetfinder.git. Specifically, in targetfinder/paper/targetfinder directory, each cell line has its own subdirectory. Furthermore, each cell line has 3 training datasets with their own subdirectories: one with features generated for the enhancer and promoter only (EP), one for promoters and extended enhancers (EEP), and one for promoters, enhancers, and the window between (EPW). For example, paper/targetfinder/HeLa-S3/output-eep contains training data for the HeLa-S3 cell line using promoter and extended enhancer features. More detailed information of the directory can be found in https://github.com/shwhalen/targetfinder.git.

# Model Training

Before training, you need to find out which cell line you are interested in. For example, if you want to find out the result of EP2vec of K562, move the ep2vec.py into paper/targetfinder/K562/output-eep and simply run 
    ```python ep2vec.py```
and you can get the AUC scores and F1 scores of 10-fold cross-validation.





