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
#################################################################
###############Function Box#####################################
##################################################################

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

def lda_params(data, label):
 #   bp()
    ##defining the class
    class0 = data[label == '0']
    class1 = data[label == '1']
 #   class0=list()
 #   class1=list()
  #  r,c=np.shape(data)
  #  for i in range(r):
   #     if (label[i] == 0):
   #         class0.append(data[i,:])
   #     if (label[i] == 1):
   #         class1.append(data[i,:])
 #   bp()
    class0=np.array(class0)
    class1=np.array(class1)
    ##calculate the mean
    mean0 = np.mean(class0, 0)
    mean1 = np.mean(class1, 0)
    ###calcvualte the within and between class variance
    S_b = np.dot((mean1 - mean0), np.transpose(mean1 - mean0))
    S_w = np.dot(np.transpose(class0 - mean0), (class0 - mean0)) + np.dot(np.transpose(class1 - mean1),
                                                                          (class1 - mean1))
    ####calcualte the matrxi whose gev you want to determine
    mat = np.dot(np.linalg.pinv(S_w), S_b)
    # eigvals, eigvecs = np.linalg.eig(mat)
    # eiglist = [(eigvals[i], eigvecs[:, i]) for i in range(len(eigvals))]

    # eiglist = sorted(eiglist, key = lambda x : x[0], reverse = True)
    # w = np.array([eiglist[i][1] for i in range(1)])
    U, s, V = np.linalg.svd(mat, full_matrices=True)
    U_reduced = U[:, : 2]
    w = U_reduced.reshape(2, data.shape[1])
    ###calcuating the threshold, the main idea of formula is project the mean of both data on lowere space and than find the distance between them and selcte
    ###the center point
    threshold = 0.5 * (np.dot(w, mean0) + np.dot(w, mean1))
    # print("threshold",threshold)
    ####here i am defining the classes that is after projecting the data where the projected data will falln and according that label it as 0 or 1 class
    if (np.mean(np.dot(w, mean0)) > np.mean(threshold)):
        cl = 0
    else:
        cl = 1
    return w, threshold, cl


def mul_norm(x, miu, cov):
    result = np.log(math.pow(np.linalg.det(cov), -0.5) /  math.pow(2 * math.pi,128/2))
    temp = x-miu
    result += (-0.5 * np.sum(np.dot(temp, np.linalg.inv(cov))*temp,axis=1))
    return np.exp(result)

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

def kmeans(data):
    #####select the random cluster center
    mean=data[np.random.choice(data.shape[0], Clusters, replace=False), :]
    classes = np.zeros((data.shape[0], 1))
    for iter in range(50):
        for i in range(data.shape[0]):
            norm=float("inf")
            for k in range(Clusters):
                dist = np.linalg.norm(mean[k, :] - data[i, :]) ###find the distance between cluster center andf point
                if (dist < norm):
                    norm = dist
                    classes[i]=k
        for j in range(Clusters):
            index = (classes[:, 0] == j)
            mean[j] = np.mean(data[index], 0)

    alpha=np.zeros((Clusters,1))
    covariance=[]
    for j in range(Clusters):
        index=((classes[:,0]==j))
        alpha[j, 0] = np.sum(index) / data.shape[0]
        covariance.append(np.diag(np.var(data[index], axis=0)))

    return mean,covariance,alpha

def gmm(data):
    ####initialise the clusters centers
    mean,covariance,alpha=kmeans(data)
    ####define the latent variable
    b=np.zeros((Clusters,data.shape[0]))
    n_iter=30 ####no of iterations so that it get properly trained

    for iter in range(n_iter):
        for j in range(Clusters):
            b[j] = alpha[j]*mul_norm(data, mean[j], covariance[j])

        b = b/b.sum(axis=0)

        for j in range(Clusters):
            mean[j] = (1/np.sum(b[j]))*np.dot(b[j],data)
            alpha[j] = np.sum(b[j]) / data.shape[0]

        for j in range(Clusters):
            covariance[j]=np.zeros((10,10))
            for i in range(data.shape[0]):
                t = (data[i] - mean[j]).reshape((10, 1))
                covariance[j] = covariance[j] + (1 / np.sum(b[j])) * b[j, i] * np.diag(np.square(t).reshape((10,)))

        loss=_compute_loss_function(data,alpha,mean,covariance,b.T)

        data_p=data[:,0:2]
        print(np.shape(data_p))
        if iter % 3 == 0:
            print("Iteration: %d Loss: %0.6f" % (iter, loss))

            pred_labels = predict(data, mean, covariance, alpha)
            plt.scatter(data[pred_labels==0,0],data[pred_labels==0,1],0.8,color='navy')
            plt.scatter(data[pred_labels == 1, 0], data[pred_labels == 1, 1], 0.8, color='red')
            plt.savefig('/home/rohitk/Desktop/MLSP/a1/Data/emotion_classification'+str(iter)+ '.png')


    return mean,covariance,alpha

def _compute_loss_function(X, pi, mu, sigma,gamma):
        N = X.shape[0]
     #   C = gamma.shape[1]
        loss = np.zeros((N, Clusters))

        for c in range(Clusters):
            dist = mvn(mu[c], sigma[c], allow_singular=True)
            loss[:, c] = gamma[:, c] * (
                        np.log(pi[c] + 0.00001) + dist.logpdf(X) - np.log(gamma[:, c] + 0.000001))
        loss = np.sum(loss)
        return loss

def predict(X,mean,variance,alpha):

    labels=np.zeros((X.shape[0],Clusters))

    for c in range(Clusters):
        kk=mvn.pdf(X,mean[c,:],variance[c])
        labels[:,c]=alpha[c]*mvn.pdf(X,mean[c,:],variance[c])

    labels=labels.argmax(1)

    return labels


######################################################################################
#####################################################################################3
#######################################################################################

temp=open('/home/rohitk/Desktop/MLSP/a2/MovieReviews1000.txt','r')
movie_rev=[x.strip() for x in temp.readlines()]
temp.close()

temp=open('/home/rohitk/Desktop/MLSP/a2/labels.txt','r')
labels=[x.strip() for x in temp.readlines()]
temp.close()


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
M = (len(movie_rev)) ####no of reviews that are there

#####so next create a matrix of size of N*M

tf_matrix=for_tf_matrix(tokenized_corpus,N,M)
#print(tf_matrix[0:9,0])
doc_vec=doc_freq(tf_matrix)
#print(doc_vec[0:9])
#print(np.shape(doc_vec))

tf_sum = np.sum(tf_matrix,axis=0)
#print(np.shape(tf_sum))
#print(tf_sum[0])
tf_matrix = np.divide(tf_matrix,tf_sum)
#print(tf_matrix[0:9,0])

tf_matrix=np.multiply(tf_matrix,doc_vec)
#print(tf_matrix[0:9,0])
##############################################tf-idf vector created######################################################
#########################################################################################################################
#########################################################################################################################
########################################################################################################################
##PCA DONE#######################
tf_matrix_after_pca = np.transpose(np.dot(PCA_transform(np.transpose(tf_matrix), 10),(tf_matrix)))
#print(np.shape(tf_matrix_after_pca))
#print(np.shape(labels))
##############################################################################################
#######modifying the labels################
labels=np.array(labels)
#print(np.shape(labels.reshape(1000,1)))
#[w, threshold, cl] = lda_params(tf_matrix_after_pca, labels.reshape(1000, ))
print(type(labels))

num_dict={0 : '0' , 1 : '1'}

def get_num(w):
    for word, index in num_dict.items():
        if w == index:
            return word

label_n=np.zeros((1000,1))

for i in range(1000):
    if (labels[i] == '0'):
        label_n[i] = 0
    if (labels[i] == '1'):
        label_n[i] =1

#print((label_n[0:5]))

[w, threshold, cl] = lda_params(tf_matrix_after_pca, labels.reshape(1000, ))
#print(np.shape(w))
data_lda = np.dot(tf_matrix_after_pca, np.transpose(w))
#print(np.shape(data_lda))
colors = ['navy', 'turquoise']
lw = 2
cl0=data_lda[labels.reshape(1000) == '0']
cl1=data_lda[labels.reshape(1000) == '1']
print(np.shape(cl0))
print(np.shape(cl1))

#plt.scatter(cl0[:,0],cl0[:,1], color='navy', alpha = 0.8)
#plt.scatter(cl1[:,0],cl1[:,1], color = 'red', alpha=0.8)
#plt.savefig('/home/rohitk/Desktop/MLSP/a1/Data/emotion_classification' + 'GMM' + str(10) + '.png')

#################################################

####tf_matrix_after_pca
global Clusters
Clusters=2

X_train=tf_matrix_after_pca
#print(np.shape(X_train))
mean,covariance,alpha=gmm(X_train)
print(np.shape(mean))
print(np.shape(covariance))
pred_labels=predict(X_train,mean,covariance,alpha)
correct=0
error=0
for i in range(1000):
    if (labels[i] == str(pred_labels[i])):
        correct= correct+1
    else:
        error = error+1
print(correct)
print(error)
#######################################################################################################################
#################################################################################################################
###########################Comparaing the performance of our results with the sklearn in built liberray
###########################################################################################################
import numpy as np
import itertools

from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

gmm=mixture.GaussianMixture(n_components=2,covariance_type='diag',max_iter=30)
clf=gmm.fit(tf_matrix_after_pca)
print(np.shape(clf.weights_))


pred_labels=predict(X_train,clf.means_,clf.covariances_,clf.weights_)
correct=0
error=0
for i in range(1000):
    if (labels[i] == str(pred_labels[i])):
        correct= correct+1
    else:
        error = error+1
print(correct)
print(error)
#########################################################################################################################
print(tf_matrix)





