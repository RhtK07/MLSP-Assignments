#######MLSP Assignmnet
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
import pdb
from sklearn import preprocessing

#####################################################################
######################################################################

#####defining the function

def PCA_transform(X, k):
    (n, m) = X.shape
    A = X - np.mean(X, 0)

    if (n > m):
        Sigma = 1.0 / n * np.dot(np.transpose(A), A)
        U, s, V = np.linalg.svd(Sigma, full_matrices=True)
        U_reduced = U[:, : k]
        Z = np.transpose(U_reduced)
        return Z
    else:
        Sigma = 1.0 / n * np.dot(A, np.transpose(A))
        U, s, V = np.linalg.svd(Sigma, full_matrices=True)
        U_reduced = np.dot(np.transpose(A), U)
        # U_red=np.linalg(U_reduced)
        U_red = U_reduced[:, : k]
        Z = np.transpose(U_red)

        return Z


def feedforwar_Relu(x,W,b):
    z=np.matmul(W,x)+b
    y=np.maximum(z,0)
    return y

def feedforward_softmax(x,W,b,label):
    no_train=len(label) ####number of training example
    z=np.matmul(W,x)+b
    z=np.exp(z)

    z_sum=np.sum(z,axis=0)
    z_mod = z / z_sum
    value=np.zeros([no_train,1])
    for i in range(no_train):
        if label[i]==0:
            value[i]=z_mod[label[i],i]
        else:
            value[i]=z_mod[label[i],i]
    loss = -np.sum(np.log(value))/no_train



    return loss,z_mod


def nn_model(Z_after_pca, W1, W2, b1, b2, label_Z):
   # pdb.set_trace()
    input=Z_after_pca.T
    a1 = feedforwar_Relu(input, W1, b1)

    loss, output = feedforward_softmax(a1, W2, b2, label_Z)
    #####output---->>>Output matrix in which probability value are there
    grad = {}  ####fprming the dictionary in which we are going to store the gradient value
    #######This thing is wrong,I have to modify this thing only
   # class_label = np.reshape(np.array([0, 1]), (2, 1))
   # da = (output - class_label) / 20
    


    dW2 = a1.dot(da.T)
    # print(dW2)
    db2 = np.sum(output, axis=1)
#    print(da.shape)
    hidden = W2.T.dot(da)
    hidden[a1 == 0] = 0
#    print(hidden.shape)
#    print(Z_after_pca.shape)
    dW1 = Z_after_pca.T.dot(hidden.T)
    db1 = np.sum(hidden, axis=1)
    #####defining the gradiaent
    grad['W2'] = dW2
    grad['W1'] = dW1
    grad['b1'] = db1
    grad['b2'] = db2

    return loss, grad
#########################################################################
################################Writing the main function

Z = np.zeros((20, 101, 101))  ###20 images with 101*101 size

label_Z = np.zeros((20, 1),dtype=int)
i = 0

path = '/home/rohitk//Desktop/MLSP/a3/Data/emotion_classification/train'
#######creating the labels
for image in os.listdir('/home/rohitk//Desktop/MLSP/a3/Data/emotion_classification/train'):
    Z[i] = plt.imread(os.path.join(path, image))
    if (image.split(".")[1] == "happy"):
        label_Z[i, :] = 1
    else:
        label_Z[i, :] = 0

    i = i + 1
#########################

Z = Z.reshape((20, 101 * 101))  ####flatten each input image

#####Normalising and scaling of the data

Z_after_pca = np.transpose(np.matmul(PCA_transform(Z, 10), np.transpose(Z)))
Z_after_pca = preprocessing.scale(preprocessing.normalize(Z_after_pca))


O = np.zeros((10, 101, 101))
label_O = np.zeros((10, 1),dtype=int)
i = 0
path = '/home/rohitk//Desktop/MLSP/a3/Data/emotion_classification/test'

for image in os.listdir('/home/rohitk//Desktop/MLSP/a3/Data/emotion_classification/test'):
    O[i] = plt.imread(os.path.join(path, image))

    if (image.split(".")[1] == "happy"):
        label_O[i, :] = 1
    else:
        label_O[i, :] = 0

    i = i + 1

O = O.reshape((10, 101 * 101))
#####Normalising and scaling of data
#O = preprocessing.scale(preprocessing.normalize(O))


O_after_pca = np.transpose(np.dot(PCA_transform(O, 10), np.transpose(O)))
O_after_pca = preprocessing.scale(preprocessing.normalize(O_after_pca))
#print(Z_after_pca)
#print(O_after_pca.shape)
####Data Preparation work is done,lets do the feed forward path
############################################################################################################
####input layer-->>>10
#####hidden layer----->>>10(ReLU)
####output------>>>>2(softmax)(cross entropy)
#std=1e-4









W1=np.random.rand(10,10)

b1=np.zeros((10,1))
W2=np.random.rand(2,10)
b2=np.zeros((2,1))
##########################################
iteration=10

for iter in range(iteration):
   # pdb.set_trace()
    loss,grad=nn_model(Z_after_pca,W1=W1,W2=W2,b1=b1,b2=b2,label_Z=label_Z)

    learning_rate=1

    W1 -= learning_rate*(grad['W1'].T)
    W2 -= learning_rate * (grad['W2'].T)
    grad['b1']=np.reshape(grad['b1'],(10,1))
    grad['b2'] = np.reshape(grad['b2'], (2, 1))
    b1 -= learning_rate * grad['b1']

    b2 -= learning_rate * grad['b2']

    print('loss function is'+str(loss))









#a1=feedforwar_Relu(Z_after_pca.T,W1,b1)
#print(a1)
#print(a1.shape)
#loss,output=feedforward_softmax(a1,W2,b2,label_Z)
####print(loss)
##############Defining theback prapogation
#####
#grad={}  ####fprming the dictionary in which we are going to store the gradient value
#class_label=np.reshape(np.array([0,1]),(2,1))
#da=(output-class_label)/20
#####print(grad['a'])
#print(a1.shape)
#print(da.shape)
#dW2=a1.dot(da.T)
#print(dW2)
#db2=np.sum(output,axis=1)
#print(da.shape)
#hidden=W2.T.dot(da)
#hidden[a1==0]=0
#print(hidden.shape)
#print(Z_after_pca.shape)
#dW1=Z_after_pca.T.dot(hidden.T)
#db1=np.sum(hidden,axis=1)
#####defining the gradiaent
#grad['W2']=dW2
#grad['W1']=dW1
#grad['b1']=db1
#grad['b2']=db2












