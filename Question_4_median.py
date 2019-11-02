from sklearn.model_selection import train_test_split
import os
import numpy as np
import scipy
import scipy.misc
import PIL
from PIL import Image
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import linalg

############################################################################################################################################
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
        Z = np.transpose(U_red)
        return Z
################################################################################################################################
#################calculate the lda parameter
def lda_params(data, label):
    ##defining the class
    class0 = data[label.reshape(label.shape[0], ) == 0]
    class1 = data[label.reshape(label.shape[0], ) == 1]
    ##calculate the mean
    mean0 = np.median(class0, 0)
    mean1 = np.median(class1, 0)
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
    U_reduced = U[:, : 1]
    w = U_reduced.reshape(1, data.shape[1])
    ###calcuating the threshold, the main idea of formula is project the mean of both data on lowere space and than find the distance between them and selcte
    ###the center point
    threshold = 0.5 * (np.dot(w, mean0) + np.dot(w, mean1))
    # print("threshold",threshold)
    ####here i am defining the classes that is after projecting the data where the projected data will falln and according that label it as 0 or 1 class
    if (np.dot(w, mean0) > threshold):
        cl = 0
    else:
        cl = 1
    return w, threshold, cl
#######################################################################################################################################
############calculate the score of the label
def calculate_score(data, label, w, threshold, cl):
    pred = np.zeros((data.shape[0], 1), np.float32)
    for i in range(data.shape[0]):
        ####storing our prediction in ecah and every entry of matrix
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
        ###comparing our prediction with the true value and defing the error
        if (pred[i, 0] != label[i]):
            error = error + 1
    return error / data.shape[0]


Z = np.zeros((20, 101, 101))
label_Z = np.zeros((20, 1))
i = 0
path = '/home/rohitk/Desktop/MLSP/a1/Data/emotion_classification/train'  ###################### point to train directory

for image in os.listdir(
        '/home/rohitk/Desktop/MLSP/a1/Data/emotion_classification/train'):  ###################### point to train directory
    Z[i] = plt.imread(os.path.join(path, image))
    if (image.split(".")[1] == "happy"):
        label_Z[i, :] = 1
    else:
        label_Z[i, :] = 0
    i = i + 1

# print(Z.shape)
Z = Z.reshape((20, 101 * 101))
# print(Z.shape)

O = np.zeros((10, 101, 101))
label_O = np.zeros((10, 1))
i = 0
path = '/home/rohitk/Desktop/MLSP/a1/Data/emotion_classification/test'  ###################### point to test directory
for image in os.listdir(
        '/home/rohitk/Desktop/MLSP/a1/Data/emotion_classification/test'):  ###################### point to test directory
    O[i] = plt.imread(os.path.join(path, image))
    if (image.split(".")[1] == "happy"):
        label_O[i, :] = 1
    else:
        label_O[i, :] = 0
    i = i + 1

# print(O.shape)
O = O.reshape((10, 101 * 101))
# print(O.shape)
y = np.ones((20, 1))

for k in range(1, 23, 3):
    Z_after_pca = np.transpose(np.dot(PCA_transform(Z, k), np.transpose(Z)))
    plt.figure()
    [w, threshold, cl] = lda_params(Z_after_pca, label_Z.reshape(20, ))
    data_lda = np.dot(Z_after_pca, np.transpose(w))
    # print(data_lda[label_Z.reshape(20,) == 0])
    colors = ['navy', 'turquoise']
    lw = 2
    for color, i, target_name in zip(colors, [0, 1], ["sad", "happy"]):
        plt.scatter(data_lda[label_Z.reshape(20, ) == i], np.zeros_like(data_lda[label_Z.reshape(20, ) == i]) + 0.0,
                    color=color, alpha=.8, label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA of dataset' + str(k))
    plt.savefig('/home/rohitk/Desktop/MLSP/a1/Data/emotion_classification/Output' + 'LDA of dataset' + str(
        k) + '.png')  ###################### point to output directory
    print("k=", k, "threshold=", threshold, "score=", (
                1 - calculate_score(np.transpose(np.dot(PCA_transform(Z, k), np.transpose(O))), label_O.reshape(10, ),
                                    w, threshold, cl)) * 100)

