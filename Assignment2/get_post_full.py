
from __future__ import division
from __future__ import print_function
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
train_dir=/home/rohitk/Desktop/MLSP/a2/speech_music_classification"






mfcc_values = {
    'wshift':0.010,
    'w_size':0.025
}

def mul_norm(x, miu, cov):
    result = math.pow(np.linalg.det(cov), -0.5) / (2 * math.pi)
    temp = x-miu
    result *= np.exp(-0.5 * np.sum(np.dot(temp, np.linalg.inv(cov))*temp,axis=1))
    return result

def parse_args(args=sys.argv):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        'n_components', type=int,
        help='number of components for GMM training')
    parser.add_argument(
        'type_covariance', type=str,
        help='enter "full" for full covariance matrix and diag for diag covariance matrix')

    
    return parser.parse_args()

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


def get_train_data(data_class):
    train_path=train_dir+"/train/"+data_class
    train_files=glob.glob(train_path+"/*.wav")
    Xtrain=[]
    for train_file in train_files:      
    #for i in range(0,10):        
        #train_file=train_files[i]
	if data_class in train_file:        
		print(train_file)
		[fs,data1]=scipy.io.wavfile.read(train_file, mmap=False)
		data=np.log10(np.abs(stft(data1, fs, 0.025, 0.010))) #np.log10
                #feature=librosa.core.stft(data1,n_fft=512, hop_length=hop, win_length=window, window='hann', center=True, dtype=np.complex128, paf_mode='reflect')           
		Xtrain.extend(data)
    Xtrain=np.array(Xtrain)
    #for i in range(Xtrain.shape[1]):
	#mean_=np.mean(Xtrain[:,i])
        #var_=np.var(Xtrain[:,i])
	#Xtrain[:,i]=(Xtrain[:,i]-mean_)/np.sqrt(var_)
    return Xtrain

def kmeans(data):
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
		index=(classes[:,0]==j)
                #index=np.transpose(index)
                #print(index)
                #print(data[classes[:,0]==j])
		mean[j]=np.mean(data[index],0)
    alpha=np.zeros((N,1))
    covariance=[]
    for j in range(N):
	index=(classes[:,0]==j)
	alpha[j,0]=np.sum(index)/data.shape[0]
	covariance.append(np.cov(np.transpose(data[index])))
        #covariance.append(np.identity(128))
        #alpha[0]=0.5
        #alpha[1]=0.5
    return mean,covariance,alpha


def gmm(data,typea):
    mean,covariance,alpha=kmeans(data)
    b=np.zeros((N,data.shape[0]))
    n_iter=30
    likelihood=np.zeros((n_iter,1))
    #print("cov",covariance)
    #print(mean,covariance,alpha)
    for iter in range(n_iter):
        for j in range(N):
                #print(np.linalg.det(covariance[j]))
        	b[j]=alpha[j]*multivariate_normal.pdf(data, mean[j], covariance[j])
        likelihood[iter]=np.sum(np.log(np.sum(b,axis=0)))
        b = b/b.sum(axis=0)
        #print("b",b[0:2,0:50])
        print("likelihood",likelihood[iter])
	for j in range(N):
		mean[j]=(1/np.sum(b[j]))*np.dot(b[j],data)
		alpha[j]=np.sum(b[j])/data.shape[0]
        print(alpha)
	#print("mean",mean[0])    
        #print("mean",mean[0])    
	for j in range(N):
		covariance[j]=np.zeros((128,128))
                #print((data[0]-mean[0]).shape)
                #print(data[0]-mean[0])
		for i in range(data.shape[0]):
                        t=(data[i]-mean[j]).reshape((128,1))
                        #print(np.dot(t,np.transpose(t)))
	        	covariance[j]=covariance[j]+b[j,i]*np.dot(t,np.transpose(t))
                #print(covariance[j])
		covariance[j]=(1/np.sum(b[j]))*covariance[j]
                covariance[j]=covariance[j]
                #print(covariance[j])
    post_dir="/home/rohitk/Desktop/MLSP/a2/final_sub"+"/1parametersf_"+str(N)
    np.save(post_dir+'/mean_'+typea,mean)
    np.save(post_dir+'/covariance_'+typea,covariance)   
    np.save(post_dir+'/alpha_'+typea,alpha)   
    np.save(post_dir+'/likelihoof_'+typea,likelihood)   
    return mean,covariance,alpha,likelihood
	
  




def test_error():

    #Xtrain=get_train_data("music")
    #mean,covariance,alpha,likelihood=gmm(Xtrain,"music")
    #print(likelihood)
    #Xtrain1=get_train_data("speech")
    #mean1,covariance1,alpha1,likelihood1=gmm(Xtrain1,"speech")
    #print(likelihood1)
    mean=np.load('/home/rohitk/Desktop/MLSP/a2/final_sub/1parametersf_'+str(N)+'/mean_music.npy')
    alpha=np.load('/home/rohitk/Desktop/MLSP/a2/final_sub/1parametersf_'+str(N)+'/alpha_music.npy')
    covariance=np.load('/home/rohitk/Desktop/MLSP/a2/final_sub/1parametersf_'+str(N)+'/covariance_music.npy')
    mean1=np.load('/home/rohitk/Desktop/MLSP/a2/final_sub/1parametersf_'+str(N)+'/mean_speech.npy')
    alpha1=np.load('/home/rohitk/Desktop/MLSP/a2/final_sub/1parametersf_'+str(N)+'/alpha_speech.npy')
    covariance1=np.load('/home/rohitk/Desktop/MLSP/a2/final_sub/1parametersf_'+str(N)+'/covariance_speech.npy')
    correct=0
    total=0
    test_dir=train_dir+"/test"
    test_files=glob.glob(test_dir+"/*.wav")
    for test_file in test_files: 
        total=total+1     
	[fs,data1]=scipy.io.wavfile.read(test_file, mmap=False)
	data=np.log10(np.abs(stft(data1, fs, 0.025, 0.010))) #np.log10
        b=np.zeros((N,data.shape[0]))
        for j in range(N):
        	b[j]=alpha[j,0]*multivariate_normal.pdf(data, mean[j], covariance[j])
        likelihoof_music=np.sum(np.log(np.sum(b,0)))
        b1=np.zeros((N,data.shape[0]))
        for j in range(N):
        	b1[j]=alpha1[j,0]*multivariate_normal.pdf(data, mean1[j], covariance1[j])
        likelihoof_speech=np.sum(np.log(np.sum(b1,0)))
        classe=test_file.split("/")[-1].split("_")[0]
        print(classe,likelihoof_music,likelihoof_speech)
        if (likelihoof_music > likelihoof_speech and classe=="music"):
            correct=correct+1
            print("correct")
        elif (likelihoof_music < likelihoof_speech and classe=="speech"):
            correct=correct+1
            print("correct")
        else:
	    print("error")
    print("accuracy",correct*100/total)
    

    return



global N
N = 2
#N=5
test_error()
Xtrain=get_train_data("music")
Xtrain_=get_train_data("speech")
#print(Xtrain.shape)
np.random.shuffle(Xtrain)
np.random.shuffle(Xtrain_)
Xtrain=Xtrain[0:15000]
Xtrain_=Xtrain_[0:15000]
# print(Xtrain_.shape)
# print(Xtrain[0])
# print(Xtrain_[0])
#mean,covariance,alpha=kmeans(Xtrain)
#mean,covariance,alpha,likelihood=gmm(Xtrain,"music")
    # print(mean)
    # print(alpha)
    # print(covariance[0].shape)
    # print(covariance[0])
    # print(likelihood)
#Xtrain=get_train_data("speech")
#mean,covariance,alpha,likelihood=gmm(Xtrain_,"speech")
    # print(mean)
    # print(alpha)
    # print(covariance[0].shape)
    # print(covariance[0])
    # print(likelihood)
