#  EP2vec

</br>EP2vec is computational framework to predict enhancer-promoter interactions by extracting “sequence embedding features”, defined as fixed-length vector representations learned from variable-length sequences, via a deep learning method in natural language processing named Paragraph Vector. In "EP2vec: a deep learning approach for extracting sequence embedding features to predict enhancer-promoter interactions", we</br>

* Extract the fasta files from bed files
* Split the sequence into k-mer words
* Generate training pairs from TargetFinder
* Use Paragraph Vector to train enhancer and promoter sequence embedding features seperately
* Identify the true enhancer-promoter interactions from other possible interactions within a topologically associating domain (TAD)
</br>

![image](http://github.com/wanwenzeng/ep2vec/raw/master/workflow.jpg)

<br>The two-stage workflow of EP2vec. Stage 1 of EP2vec is unsupervised feature extraction which transforms enhancer sequences and promoter sequences in a cell line into sequence embedding features separately. Given a set of all known enhancers or promoters in a cell line, we first split all the sequences into k-mer words with stride s=1 and assign a unique ID to each of them. Regarding the preprocessed sequences as sentences, we embed each sentence to a vector by Paragraph Vector. Concretely, we use vectors of words in a context with the sentence vector to predict the next word in the context using softmax classifier. After training converges, we get embedding vectors for words and all sentences, where the vectors for sentences are exactly the sequence embedding features that we need. Note that in sentence ID, SEQUENCE is a placeholder for ENHANCER or PROMOTER, and  is the total number of enhancers or promoters in a cell line. Stage 2 is supervised learning for predicting EPIs. Given a pair of sequences, namely an enhancer sequence and a promoter sequence, we represent the two sequences using the pre-trained vectors and then concatenate them to obtain the feature representation. Lastly, we train a Gradient Boosted Gradient Trees classifier to predict whether this pair is a true EPI.<br>




##  Training Data

</br>EP2vec uses the same training data as TargetFinder, where interacting enhancer-promoter pairs are annotated using high-resolution genome-wide Hi-C data (Rao et al., 2014). Labeled training datasets used in the TargetFinder are available in https://github.com/shwhalen/targetfinder.git. Specifically, in targetfinder/paper/targetfinder directory, each cell line has its own subdirectory. Furthermore, each cell line has 3 training datasets with their own subdirectories: one with features generated for the enhancer and promoter only (EP), one for promoters and extended enhancers (EEP), and one for promoters, enhancers, and the window between (EPW). For example, paper/targetfinder/HeLa-S3/output-eep contains training data for the HeLa-S3 cell line using promoter and extended enhancer features. More detailed information of the directory can be found in https://github.com/shwhalen/targetfinder.git.</br>

## Model Training

</br>Before training, you need to find out which cell line you are interested in. For example, if you want to find out the result of EP2vec of K562, move the ep2vec.py into paper/targetfinder/K562/output-eep. ep2vec.py accepts 3 parameters, the length of k-mer, the length of stride, the the dimension of embedding vector. If you want to get the result of 6-mer with stride 1 and embedding dimension 100, simply run</br>
```
python ep2vec.py 6 1 100
```
</br>
and you can get the AUC scores and F1 scores of 10-fold cross-validation. More detailed of the source code can be found in ep2vec.py.</br>


## Dependency

</br>EP2vec requires:

* Python
* gensim
* scikit-learn  
* bedtools

</br>




