import os
import random
import numpy
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.cross_validation import StratifiedKFold, cross_val_score
import sys




cv = StratifiedKFold(y = labels, n_folds = 10, shuffle = True, random_state = 0)
f1        = []
auc       = []
aupr      = []
for train,test in cv:
        estimator = GradientBoostingClassifier(n_estimators = 4000, learning_rate = 0.001, max_depth = 25, max_features = 'log2', random_state = 0)
        estimator.fit(arrays[train,:], labels[train])
        sort_index = [i[0] for i in sorted(enumerate(estimator.feature_importances_), key=lambda x:x[1])]
        array = arrays[:,sort_index[-16:]]
        estimator.fit(array[train,:], labels[train])
        y_pred = estimator.predict(array[test,:])
        y_prob = estimator.predict_proba(array[test,:])
        f1.append(metrics.f1_score(labels[test],y_pred))
        fpr, tpr,th = metrics.roc_curve(labels[test],y_prob[:,1], pos_label=1)
        auc.append(metrics.auc(fpr, tpr))
        aupr.append(metrics.average_precision_score(labels[test],y_prob[:,1]))


print 'TargetFinder,feature selection:'
print numpy.mean(f1),numpy.std(f1)
print numpy.mean(auc),numpy.std(auc)
print numpy.mean(aupr),numpy.std(aupr)
