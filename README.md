#  EP2vec

</br>EP2vec is computational framework to predict enhancer-promoter interactions by extracting “sequence embedding features”, defined as fixed-length vector representations learned from variable-length sequences, via a deep learning method in natural language processing named Paragraph Vector. In "EP2vec: a deep learning approach for extracting sequence embedding features to predict enhancer-promoter interactions", we</br>

* Extract the fasta files from bed files
* Split the sequence into k-mer words
* Generate training pairs from TargetFinder
* Use Paragraph Vector to train enhancer and promoter sequence embedding features seperately
* Identify the true enhancer-promoter interactions from other possible interactions within a topologically associating domain (TAD)
</br>

![Alt Text](http://github.com/wanwenzeng/ep2vec/raw/master/workflow.jpg)


##  Training Data

</br>EP2vec uses the same training data as TargetFinder, where interacting enhancer-promoter pairs are annotated using high-resolution genome-wide Hi-C data (Rao et al., 2014). Labeled training datasets used in the TargetFinder are available in https://github.com/shwhalen/targetfinder.git. Specifically, in targetfinder/paper/targetfinder directory, each cell line has its own subdirectory. Furthermore, each cell line has 3 training datasets with their own subdirectories: one with features generated for the enhancer and promoter only (EP), one for promoters and extended enhancers (EEP), and one for promoters, enhancers, and the window between (EPW). For example, paper/targetfinder/HeLa-S3/output-eep contains training data for the HeLa-S3 cell line using promoter and extended enhancer features. More detailed information of the directory can be found in https://github.com/shwhalen/targetfinder.git.</br>

## Model Training

</br>Before training, you need to find out which cell line you are interested in. For example, if you want to find out the result of EP2vec of K562, move the ep2vec.py into paper/targetfinder/K562/output-eep. ep2vec.py accepts 3 parameters, the length of k-mer, the length of stride, the the dimension of embedding vector. If you want to get the result of 6-mer with stride 1 and embedding dimension 100, simply run</br>
```
python ep2vec.py 6 1 100
```
</br>
and you can get the AUC scores and F1 scores of 10-fold cross-validation. More detailed of the source code can be found in ep2vec.py.</br>

## Result

<br/>
The F1 scores of EP2vec and TargetFinder:

|cell lines   |EP2vec(E/P)  |TargetFinder(E/P)|TargetFinder(EE/P)|TargetFinder(E/P/W)|
|-------------|-------------|-----------------|------------------|-------------------|
|K562         |0.88         |0.61             |0.81              |0.85               |
|GM12878      |0.86         |0.48             |0.78              |0.83               |
|HUVEC        |0.87         |0.59             |0.81              |0.84               |
|HeLa-S3      |0.87         |0.48             |0.77              |0.71               |
|IMR90        |0.91         |0.61             |0.87              |0.83               |
|NHEK         |0.92         |0.59             |0.90              |0.83               |


## Dependency

</br>EP2vec requires:

* Python
* gensim
* scikit-learn  
* bedtools

</br>




