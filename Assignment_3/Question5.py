from __future__ import division
from __future__ import print_function
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
from scipy.fftpack import dct
import os
import scipy
import scipy.misc
from pdb import set_trace as bp
import scipy
import scipy.io.wavfile
import os
import sys
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import glob
import numpy as np
import librosa
import math
from scipy.stats import multivariate_normal as mvn
from sklearn.model_selection import train_test_split
import os
import numpy as np
import scipy
import PIL
from PIL import Image
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import linalg
import six.moves.cPickle as pickle
import gzip
import numpy
from sklearn import svm
from sklearn.model_selection import learning_curve, GridSearchCV
#from sklearn.grid_search import GridSearchCV

###############################################################################################################


def tokenize_corpus(corpus):
    tokens = [x.split() for x in corpus]
    return tokens

def creat_vocab(token_corpus):
    vocabulary = []
    for sentence in token_corpus:
        for token in sentence:
            if token not in vocabulary:
                vocabulary.append(token)

    word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
    idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}
    vocabulary_size = len(vocabulary)

    return vocabulary,vocabulary_size,word2idx,idx2word


def get_key(w):
    for word, index in word2idx.items():
        if w == word:
            return index


def for_tf_matrix(token_corpus,N,M):
    ###N=no fo words in dict
    ####M no of reviews
    tf_matrix = np.zeros((N, M))
    col=0  ####to make count of no of column we are going to access
    for sentence in token_corpus:
        for token in sentence:
            x=get_key(token)
            tf_matrix[x,col] += 1
        col += 1
    return tf_matrix

def doc_freq(tf_matrix):
    N ,M =np.shape(tf_matrix)
    doc_vec= np.count_nonzero(tf_matrix, axis=0)
#    print(doc_vec[0:5])
#    print(M)
    doc_vec = np.log(M*(1./doc_vec))
#    print(doc_vec[0:5])

    return doc_vec

def PCA_transform(X, k):
    ####Define the shape of matrix [eg*size of X]
    (n, m) = X.shape
    ####subtract the mean from the data
    A = X - np.mean(X, 0)
    #####two cases above one is con one and second one we will use
    if (n > m):
        Sigma = 1.0 / n * np.dot(np.transpose(A), A)
        U, s, V = np.linalg.svd(Sigma, full_matrices=True)
        U_reduced = U[:, : k]
        Z = np.transpose(U_reduced)
        return Z
    else:
        ###calculate the covariance matrix get matrx of [eg*eg]
        Sigma = 1.0 / n * np.dot(A, np.transpose(A))
        #####do svd
        U, s, V = np.linalg.svd(Sigma, full_matrices=True)
        ####from the 20*20 we get the approx eigenvector of 10k*10k something,
        ####since we are not doing svd here thats why data matrxi we will obtain from here its covariance matric will not give ous idenetity
        U_reduced = np.dot(np.transpose(A), U)
        # U_red=np.linalg(U_reduced)
        ####calcualte the best k outoff them
        U_red = U_reduced[:, : k]
   #     print(np.shape(U_red))
        Z = np.transpose(U_red)
        return Z

def stft(x, fs, framesz, hop):
    """x is the time-domain signal
    fs is the sampling frequency
    framesz is the frame size, in seconds
    hop is the the time between the start of consecutive frames, in seconds
    """
    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    w = scipy.hamming(framesamp)
    X = scipy.array([scipy.fft(w*x[i:i+framesamp],256)
                     for i in range(0, len(x)-framesamp, hopsamp)])
    X=X[:,0:128]
    return X

def calculate_score(data, label, w, threshold, cl):
    pred = np.zeros((data.shape[0], 1), np.float32)
    for i in range(data.shape[0]):
        if (np.dot(w, data[i, :].reshape(data.shape[1], 1)) > threshold):
            pred[i, 0] = cl
        else:
            if (cl == 0):
                pred[i, 0] = 1
            else:
                pred[i, 0] = 0
    # print(pred)
    error = 0.00
    for i in range(data.shape[0]):
        if (pred[i, 0] != label[i]):
            error = error + 1
    return error / data.shape[0]

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

temp=open('/home/rohitk/Desktop/MLSP/a3/train.txt','r')
train_txt=[x.strip() for x in temp.readlines()]
temp.close()

temp=open('/home/rohitk/Desktop/MLSP/a3/test.txt','r')
test_txt=[x.strip() for x in temp.readlines()]
temp.close()

temp=open('/home/rohitk/Desktop/MLSP/a3/labels_train.txt','r')
labels_train=[x.strip() for x in temp.readlines()]
temp.close()

temp=open('/home/rohitk/Desktop/MLSP/a3/labels_test.txt','r')
labels_test=[x.strip() for x in temp.readlines()]
temp.close()

temp=open('/home/rohitk/Desktop/MLSP/a3/text.txt','r')
movie_rev=[x.strip() for x in temp.readlines()]
temp.close()

#################prepare tf-idf feature for train dataset######################

tokenized_corpus = tokenize_corpus(movie_rev)
#print(tokenized_corpus)
##vocabulary is what its name say
##N is number of words in vocab
###wordidx is word to itsz coreesponding id
#####idxtoword is idenetity to word
vocabulary ,N ,word2idx , idx2word  = creat_vocab(tokenized_corpus)
##print(tokenized_corpus)
##print(word2idx)
#print(vocabulary)
M1 = (len(train_txt)) ####no of reviews that are there in training dataset
M2 = (len(test_txt))  ######no of review are there in test dataset

#####so next create a matrix of size of N*M
tokenized_corpus_train = tokenize_corpus(train_txt)

tokenized_corpus_test = tokenize_corpus(test_txt)

tf_matrix_train=for_tf_matrix(tokenized_corpus_train,N,M1) ####tf matrix for training dataset
tf_matrix_test=for_tf_matrix(tokenized_corpus_test,N,M2)    #####tf matrix for testing dataset
#print(tf_matrix[0:9,0])
doc_vec_train=doc_freq(tf_matrix_train)
doc_vec_test=doc_freq(tf_matrix_test)
#print(doc_vec[0:9])
#print(np.shape(doc_vec))

tf_sum_train = np.sum(tf_matrix_train,axis=0)
tf_sum_test = np.sum(tf_matrix_test,axis=0)
#print(np.shape(tf_sum))
#print(tf_sum[0])
tf_matrix_train = np.divide(tf_matrix_train,tf_sum_train)
tf_matrix_test = np.divide(tf_matrix_test,tf_sum_test)
#print(tf_matrix[0:9,0])

tf_matrix_train=np.multiply(tf_matrix_train,doc_vec_train)
tf_matrix_test=np.multiply(tf_matrix_test,doc_vec_test)
#print(tf_matrix[0:9,0])
#########################################################################################################################
#########################################################################################################################
###########################################################################################################################

tf_matrix_after_pca_train = np.transpose(np.dot(PCA_transform(np.transpose(tf_matrix_train), 10),(tf_matrix_train)))
tf_matrix_after_pca_test = np.transpose(np.dot(PCA_transform(np.transpose(tf_matrix_test), 10),(tf_matrix_test)))


label_n_train=np.zeros((700,1))

for i in range(700):
    if (labels_train[i] == '0'):
        label_n_train[i] = 0
    if (labels_train[i] == '1'):
        label_n_train[i] =1

label_n_test=np.zeros((300,1))

for i in range(300):
    if (labels_test[i] == '0'):
        label_n_test[i] = 0
    if (labels_test[i] == '1'):
        label_n_test[i] =1

#print(np.shape(tf_matrix_after_pca_test))
#print(np.shape(tf_matrix_after_pca_train))
#print(labels_test[0:5])
#######################################################################################################################
########################################################################################################################

maxq = 0
varying_C = []
C = range(1, 101, 1)
best_c_poly=0
accuracy_poly=0
for c in C:
    clf_polynomial = svm.SVC(C=c, cache_size=200, class_weight=None, coef0=0.0,
                             decision_function_shape='ovo', degree=5, gamma='auto', kernel='poly',
                             max_iter=100, probability=False, random_state=None, shrinking=True,
                             tol=0.001, verbose=False)
    clf_polynomial.fit(tf_matrix_after_pca_train, label_n_train)
    accuracy_poly = clf_polynomial.score(tf_matrix_after_pca_test, label_n_test) * 100
    varying_C = varying_C + [accuracy_poly]
    if (accuracy_poly > maxq):
        best_c_poly = c
        maxq = accuracy_poly
#print("1", 10, best_c_poly, accuracy_poly)
r1,c1=(np.shape(clf_polynomial.support_vectors_))
######################################################################################################################
#######################################################################################################################

maxq=0
varying_C=[]
best_c_lin=0
accuracy_lin=0

C = range(1, 101, 1)
for c in C:
#    clf_linear = svm.LinearSVC(C=c, class_weight=None, dual=True, fit_intercept=True,
#                                   intercept_scaling=1, loss='squared_hinge', max_iter=100,
#                                   multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
#                                   verbose=0)
    clf_linear = svm.SVC(C=c, cache_size=200, class_weight=None, coef0=0.0,
                      decision_function_shape='ovo', degree=5, gamma='auto', kernel='linear',
                      max_iter=100, probability=False, random_state=None, shrinking=True,
                      tol=0.001, verbose=False)
    clf_linear.fit(tf_matrix_after_pca_train, label_n_train)
    accuracy_lin = clf_linear.score(tf_matrix_after_pca_test, label_n_test) * 100
    varying_C = varying_C + [accuracy_lin]
    if (accuracy_lin > maxq):
        best_c_lin = c
        maxq = accuracy_lin
r2,c2=(np.shape(clf_linear.support_vectors_))
############################################################################################
#############################################################################################
maxq = 0
varying_C = []
accuracy_rbf=0
best_c_rbf=0
C = range(1, 101, 1)
for c in C:
    clf_rbf = svm.SVC(C=c, cache_size=200, class_weight=None, coef0=0.0,
                          decision_function_shape='ovo', degree=5, gamma='auto', kernel='rbf',
                          max_iter=100, probability=False, random_state=None, shrinking=True,
                          tol=0.001, verbose=False)
    clf_rbf.fit(tf_matrix_after_pca_train, label_n_train)
#    print(np.shape(clf_rbf.support_vectors_))
#    cc=clf_rbf.support_vectors_
    accuracy_rbf = clf_rbf.score(tf_matrix_after_pca_test,label_n_test) * 100
    varying_C = varying_C + [accuracy_rbf]
    if (accuracy_rbf > maxq):
        best_c_rbf = c
        maxq = accuracy_rbf

r3,c3=(np.shape(clf_rbf.support_vectors_))
print(r1)
print(r2)
print(r3)
print("1", 10, best_c_poly, accuracy_poly)
print("2", 10, best_c_lin, accuracy_lin)
print("3", 10, best_c_rbf, accuracy_rbf)