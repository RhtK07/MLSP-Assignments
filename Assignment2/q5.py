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


#########################################################################################
def kmeans(data):
    N=2
    mean=data[np.random.choice(data.shape[0], N, replace=False), :]
    classes=np.zeros((data.shape[0],1))
    for iter in range(50):
        for i in range(data.shape[0]):
            norm=float("inf")
            for k in range(N):
                dist=np.linalg.norm(mean[k,:]-data[i,:])
                if (dist<norm):
                    norm=dist
                    classes[i]=k
        for j in range(N):
            index = (classes[:, 0] == j)
            mean[j]=np.mean(data[index],0)
    alpha=np.zeros((N,1))
    covariance=[]
    for j in range(N):
        index=(classes[:,0]==j)
        alpha[j,0]=np.sum(index)/data.shape[0]
        ####why two times diag, the first diag will first convert it into vector or take only the diag element and than convert it into diag matrix
        ####other one directly convert to matrix
        #covariance.append(np.diag(np.diag(np.cov(np.transpose(data[index])))))
        covariance.append(np.diag(np.var(data[index], axis=0)))
      #  covariance.append(np.cov(np.transpose(data[index])))

    return mean,covariance,alpha
###############################################################################################

from scipy.stats import multivariate_normal as mvn
#######################################################################################
def E_step(mu,covar,pi,X,N=2):
    ######consider this em algo as e and m seperately and the main function of e is that to generate the posterior probabilites
    #####that is to determine p(z|x)p(x)=p(z)p(x|z) for each data point
    ####where p(x)=avg_over_z(p(z)p(x|z))
    ####where p(x|z) is gaussian
    bp()
    r,c=np.shape(X)
    gamma=np.zeros((N,r))

#    mu = mu if _initial_means is None else _initial_means
#    pi = pi if _initial_pi is None else _initial_pi
#    sigma = sigma if _initial_cov is None else _initial_cov
    lld=0
    for i in range(N):
        gamma[i,:]=pi[i]*mul_norm(X, mu[i], covar[i])

    lld=np.sum(np.log(np.sum(gamma,axis=0,dtype=np.float128),dtype=np.float128))
    gamma_norm= np.sum(gamma, axis=0)[:,np.newaxis]
#    print(np.shape(gamma_norm))
#    print(np.shape(gamma[i,:]))

    gamma /= gamma_norm.T

    return gamma,lld

def M_step(X,gamma):
  #  bp()
    N=X.shape[0] ####no of elements
    C=gamma.shape[0]
    d=X.shape[1]

    mu=np.zeros((d,C))
    var=np.zeros((C,d,d))
    pi=np.zeros((C,1))

    for j in range(C):
        mu[:,j] = (1 / np.sum(gamma[j,:])) * np.dot(gamma[j], X)
        pi[j] = np.sum(gamma[j,:])/N
    for j in range(C):
        for i in range(N):
            t = (X[i,:] - mu[:,j]).reshape((32, 1))
       #     var[j] = var[j] + (1 / np.sum(gamma[j,:])) * gamma[j, i] * np.diag(np.square(t).reshape((32,)))
            var[j] = var[j] + gamma[j, i] * np.dot(t, np.transpose(t))
        var[j] = (1 / np.sum(gamma[j,:])) * var[j]

    return pi, mu.T, var


def _compute_liklihood(X, pi, mu, sigma, gamma):
    ####COMPUTE THE loss
    # bp()
    N = X.shape[0]
    C = gamma.shape[0]
    loss = np.zeros((N, C))

    for c in range(C):
        dist = mvn(mu[:, c], sigma[c], allow_singular=True)
        loss[:, c] = gamma[c, :].T * (np.log(pi[c] + 0.00001)) + dist.logpdf(X) - np.log(gamma[c, :] + 0.000001)
    loss = np.sum(loss)
    return loss



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



#plt.scatter(cl0[:,0],cl0[:,1], color='navy', alpha = 0.8)
#plt.scatter(cl1[:,0],cl1[:,1], color = 'red', alpha=0.8)
#plt.savefig('/home/rohitk/Desktop/MLSP/a1/Data/emotion_classification' + 'GMM' + str(10) + '.png')

#################################################

####tf_matrix_after_pca
global Clusters
Clusters=2

X_train=tf_matrix_after_pca
#print(np.shape(X_train))
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
#################################################################





