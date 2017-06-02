#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional
from keras.layers import Convolution1D, MaxPooling1D, Reshape, Permute
from keras.layers import Activation, Merge, BatchNormalization
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
np.random.seed(47)

import sys

i = int(sys.argv[1])
names = ['GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK']
name = names[i]
print 'Experiment on %s dataset' % name

print 'Loading seq data...'
pos_enhancers = open('../data/%s/pos_enhancer.fa' % name, 'r').readlines()[1::2]
pos_promoters = open('../data/%s/pos_promoter.fa' % name, 'r').readlines()[1::2]
neg_enhancers = open('../data/%s/neg_enhancer.fa' % name, 'r').readlines()[1::2]
neg_promoters = open('../data/%s/neg_promoter.fa' % name, 'r').readlines()[1::2]
MAX_LEN_en = len(pos_enhancers[0][:-1])
MAX_LEN_pr = len(pos_promoters[0][:-1])
pos_enhancers = [' '.join(enhancer[:-1].lower()) for enhancer in pos_enhancers]
neg_enhancers = [' '.join(enhancer[:-1].lower()) for enhancer in neg_enhancers]
pos_promoters = [' '.join(promoter[:-1].lower()) for promoter in pos_promoters]
neg_promoters = [' '.join(promoter[:-1].lower()) for promoter in neg_promoters]
enhancers = pos_enhancers + neg_enhancers
promoters = pos_promoters + neg_promoters
assert len(pos_enhancers) == len(pos_promoters)
assert len(neg_enhancers) == len(neg_promoters)
y = np.array([1] * len(pos_enhancers) + [0] * len(neg_enhancers))

print 'Tokenizing seqs...'
NB_WORDS = 5
tokenizer = Tokenizer(nb_words=NB_WORDS)
tokenizer.fit_on_texts(enhancers)
sequences = tokenizer.texts_to_sequences(enhancers)
X_en = pad_sequences(sequences, maxlen=MAX_LEN_en)
sequences = tokenizer.texts_to_sequences(promoters)
X_pr = pad_sequences(sequences, maxlen=MAX_LEN_pr)

acgt_index = tokenizer.word_index
print 'Found %s unique tokens.' % len(acgt_index)

print 'Spliting train, valid, test parts...'
n = len(y)
indices = np.arange(n)
np.random.shuffle(indices)
X_en = X_en[indices]
X_pr = X_pr[indices]
y = y[indices]

f1 =[]
auc =[]
aupr =[]
for fold in range(10):
	print 'fold %d' % fold
	kf = StratifiedKFold(n_splits=10)
	x = 0
	for train_index, test_index in kf.split(X_en, y):
	    if x == fold:
	        break
	    x += 1
	train_l = len(train_index)
	train_l = int(train_l*85/90.)
	X_en_train = X_en[train_index[:train_l]]
	X_pr_train = X_pr[train_index[:train_l]]
	y_train = y[train_index[:train_l]]
	X_en_valid = X_en[train_index[train_l:]]
	X_pr_valid = X_pr[train_index[train_l:]]
	y_valid = y[train_index[train_l:]]
	X_en_test = X_en[test_index]
	X_pr_test = X_pr[test_index]
	y_test = y[test_index]
	
	embedding_vector_length = 4
	nb_words = min(NB_WORDS, len(acgt_index)) # kmer_index starting from 1
	print('Building model...')
	print 'fix embedding layer with one-hot vectors'
	acgt2vec={'a': np.array([1, 0, 0, 0], dtype='float32'),
	          'c': np.array([0, 1, 0, 0], dtype='float32'),
	          'g': np.array([0, 0, 1, 0], dtype='float32'),
	          't': np.array([0, 0, 0, 1], dtype='float32'),
		  'n': np.array([0, 0, 0, 0], dtype='float32')}
	embedding_matrix = np.zeros((nb_words+1, embedding_vector_length))
	for acgt, i in acgt_index.items():
	    if i > NB_WORDS:
	        continue
	    vector = acgt2vec.get(acgt)
	    if vector is not None:
	        embedding_matrix[i] = vector
	conv_enhancer_seq = Sequential()
	conv_enhancer_seq.add(Embedding(nb_words+1,
	                    embedding_vector_length,
	                    weights=[embedding_matrix],
	                    input_length=MAX_LEN_en,
	                    trainable=False))
	conv_enhancer_seq.add(Convolution1D(1024, 40, activation='relu'))
	conv_enhancer_seq.add(MaxPooling1D(20, 20))
	conv_enhancer_seq.add(Reshape((1024, -1)))
	#print(conv_enhancer_seq.summary())
	conv_promoter_seq = Sequential()
	conv_promoter_seq.add(Embedding(nb_words+1,
	                    embedding_vector_length,
	                    weights=[embedding_matrix],
	                    input_length=MAX_LEN_pr,
	                    trainable=False))
	conv_promoter_seq.add(Convolution1D(1024, 40, activation='relu'))
	conv_promoter_seq.add(MaxPooling1D(20, 20))
	conv_promoter_seq.add(Reshape((1024, -1)))
	#print(conv_promoter_seq.summary())
	merged = Sequential()
	merged.add(Merge([conv_enhancer_seq, conv_promoter_seq], mode='concat'))
	merged.add(Permute((2, 1)))
	merged.add(BatchNormalization())
	merged.add(Dropout(0.5))
	merged.add(Bidirectional(LSTM(100, return_sequences=False), merge_mode='concat'))
	merged.add(BatchNormalization())
	merged.add(Dropout(0.5))
	#print(merged.summary())
	model = Sequential()
	model.add(merged)
	model.add(Dense(925))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=1e-5), metrics=['accuracy'])
	#print(model.summary())
	
	test = False
	if not test:
	    checkpointer = ModelCheckpoint(filepath="./model/%s_bestmodel_fold%d.h5"
	                                   % (name, fold), verbose=1, save_best_only=True)
	    earlystopper = EarlyStopping(monitor='val_loss', patience=50, verbose=1)
	
	    print 'Training model...'
	    model.fit([X_en_train, X_pr_train], y_train, nb_epoch=60, batch_size=100, 
		      shuffle=True,
	              validation_data=([X_en_valid, X_pr_valid], y_valid),
	              callbacks=[checkpointer,earlystopper],
	              verbose=1)
	
	print 'Testing model...'
	model.load_weights('./model/%s_bestmodel_fold%d.h5'% (name, fold))
	tresults = model.evaluate([X_en_test, X_pr_test], y_test, show_accuracy=True)
	print tresults
	y_pred = model.predict([X_en_test, X_pr_test], 100, verbose=1)
	print 'Calculating AUC...'
	f1.append(metrics.f1_score(y_test, y_pred>0.5))
	auc.append(metrics.roc_auc_score(y_test, y_pred))
	aupr.append(metrics.average_precision_score(y_test, y_pred))
	
print f1, auc, aupr
print np.mean(f1),np.std(f1)
print np.mean(auc),np.std(auc)
print np.mean(aupr),np.std(aupr)
