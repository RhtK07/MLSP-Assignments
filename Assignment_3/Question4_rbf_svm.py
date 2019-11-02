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


Z = np.zeros((20, 101, 101))
label_Z = np.zeros((20, 1))
i = 0
path='/home/rohitk/Desktop/MLSP/a3/Data/emotion_classification/train'

for image in os.listdir('/home/rohitk/Desktop/MLSP/a3/Data/emotion_classification/train'):
    Z[i]=plt.imread(os.path.join(path, image))
    if (image.split(".")[1]=="happy"):
        label_Z[i,:]=1
    else:
        label_Z[i,:]=0

    i = i+1

# print(Z.shape)
Z = Z.reshape((20, 101 * 101))
# print(Z.shape)

O = np.zeros((10, 101, 101))
label_O = np.zeros((10, 1))
i = 0

path='/home/rohitk/Desktop/MLSP/a3/Data/emotion_classification/test'

for image in os.listdir('/home/rohitk/Desktop/MLSP/a3/Data/emotion_classification/test'):
    O[i]=plt.imread(os.path.join(path, image))

    if (image.split(".")[1]=="happy"):
        label_O[i,:]=1
    else:
        label_O[i,:]=0

    i = i+1
########################################################################################################################
# print(O.shape)
O = O.reshape((10, 101 * 101))
# print(O.shape)
y = np.ones((20, 1))

maxq = 0
varying_C_18 = []
for k in range(18, 19):
    Z_after_pca = np.transpose(np.dot(PCA_transform(Z, k), np.transpose(Z)))
    O_after_pca = np.transpose(np.dot(PCA_transform(Z, k), np.transpose(O)))
    C = range(1, 101, 1)
    for c in C:
        clf_rbf = svm.SVC(C=c, cache_size=200, class_weight=None, coef0=0.0,
                          decision_function_shape='ovo', degree=5, gamma='auto', kernel='rbf',
                          max_iter=100, probability=False, random_state=None, shrinking=True,
                          tol=0.001, verbose=False)
        clf_rbf.fit(Z_after_pca, label_Z)
        accuracy = clf_rbf.score(O_after_pca, label_O) * 100
        varying_C_18 = varying_C_18 + [accuracy]
        if (accuracy > maxq):
            best_c = c
            maxq = accuracy
        print("1", k, c, accuracy)
########################################################################################################################

varying_k_67 = []
for k in range(1, 101, 1):
    Z_after_pca = np.transpose(np.dot(PCA_transform(Z, k), np.transpose(Z)))
    O_after_pca = np.transpose(np.dot(PCA_transform(Z, k), np.transpose(O)))
    c = best_c
    clf_rbf = svm.SVC(C=c, cache_size=200, class_weight=None, coef0=0.0,
                      decision_function_shape='ovo', degree=5, gamma='auto', kernel='rbf',
                      max_iter=100, probability=False, random_state=None, shrinking=True,
                      tol=0.001, verbose=False)
    clf_rbf.fit(Z_after_pca, label_Z)
    accuracy = clf_rbf.score(O_after_pca, label_O) * 100
    varying_k_67 = varying_k_67 + [accuracy]
    print("2", k, c, accuracy)
###################################################################################
plt.figure()
plt.plot(range(1, 101), varying_C_18, label='PCA dim=18', marker='.',
         markersize=5, color='b')
plt.ylabel('Test accuracy of rbf SVM classifier with varying C')
plt.xlabel('range of Cost C')
plt.legend()
plt.title('Test accuracy of rbf SVM classifier with varying C')
plt.savefig('/home/rohitk/Desktop/MLSP/a3/' + 'SVM_rbf_C_vary' + '.png')  ###################### point to output directory

plt.figure()
plt.plot(range(1, 101), varying_k_67, label="fixed cost C=" + str(best_c), marker='.',
         markersize=5, color='b')
plt.ylabel('Test accuracy of rbf SVM classifier with varying k(dim) in PCA')
plt.xlabel('range of k(dim) in PCA')
plt.legend()
plt.title('Test Accuracy of rbf SVM classifier with varying k in pca')
plt.savefig('/home/rohitk/Desktop/MLSP/a3/' + 'SVM_rbf_k_vary' + '.png')  ###################### point to output directory




