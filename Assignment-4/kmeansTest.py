import numpy as np
from data_loader import toy_dataset, load_digits
from kmeans import KMeans, KMeansClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from utils import Figure

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors

        Return new image from the image by replacing each RGB value in image with nearest code vectors (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    # TODO
    # - comment/remove the exception
    # - implement the function

    # DONOT CHANGE CODE ABOVE THIS LINE
    
    N, M, C = image.shape
    data = image.reshape(N * M, C)
    l2 = np.sum(((data - np.expand_dims(code_vectors, axis=1)) ** 2), axis=2)
    r = np.argmin(l2, axis=0)
    return code_vectors[r].reshape(N, M, C)
    
    # DONOT CHANGE CODE BELOW THIS LINE


################################################################################
# KMeans on 2D toy dataset
# The dataset is generated from N gaussian distributions equally spaced on N radius circle.
# Here, N=4
# KMeans on this dataset should be able to identify the 4 clusters almost clearly.
################################################################################


def kmeans_toy():
    x, y = toy_dataset(4)
    fig = Figure()
    fig.ax.scatter(x[:, 0], x[:, 1], c=y)
    fig.savefig('plots/toy_dataset_real_labels.png')

    fig.ax.scatter(x[:, 0], x[:, 1])
    fig.savefig('plots/toy_dataset.png')
    n_cluster = 4
    k_means = KMeans(n_cluster=n_cluster, max_iter=100, e=1e-8)
    centroids, membership, i = k_means.fit(x)

    assert centroids.shape == (n_cluster, 2), \
        ('centroids for toy dataset should be numpy array of size {} X 2'
            .format(n_cluster))

    assert membership.shape == (50 * n_cluster,), \
        'membership for toy dataset should be a vector of size 200'

    assert type(i) == int and i > 0,  \
        'Number of updates for toy datasets should be integer and positive'

    print('[success] : kmeans clustering done on toy dataset')
    print('Toy dataset K means clustering converged in {} steps'.format(i))

    fig = Figure()
    fig.ax.scatter(x[:, 0], x[:, 1], c=membership)
    fig.ax.scatter(centroids[:, 0], centroids[:, 1], c='red')
    fig.savefig('plots/toy_dataset_predicted_labels.png')

    np.savez('results/k_means_toy.npz',
             centroids=centroids, step=i, membership=membership, y=y)

################################################################################
# KMeans for image compression
# Here we use k-means for compressing an image
# We load an image 'baboon.tiff',  scale it to [0,1] and compress it.
# The problem can be rephrased as --- "each pixel is a 3-D data point (RGB) and we want to map each point to N points or N clusters.
################################################################################


def kmeans_image_compression():
    im = plt.imread('baboon.tiff')
    N, M = im.shape[:2]
    im = im / 255

    # convert to RGB array
    data = im.reshape(N * M, 3)

    k_means = KMeans(n_cluster=16, max_iter=100, e=1e-6)
    centroids, _, i = k_means.fit(data)

    print('RGB centroids computed in {} iteration'.format(i))
    new_im = transform_image(im, centroids)

    assert new_im.shape == im.shape, \
        'Shape of transformed image should be same as image'

    mse = np.sum((im - new_im)**2) / (N * M)
    print('Mean square error per pixel is {}'.format(mse))
    plt.imsave('plots/compressed_baboon.png', new_im)

    np.savez('results/k_means_compression.npz', im=im, centroids=centroids,
             step=i, new_image=new_im, pixel_error=mse)


################################################################################
# Kmeans for classification
# Here we use k-means for classifying digits
# We find N clusters in the data and label each cluster with the maximal class that belongs to that cluster.
# Test samples are labeled based on which cluster they belong to
################################################################################

def kmeans_classification():
    x_train, x_test, y_train, y_test = load_digits()

    # plot some train data
    N = 25
    l = int(np.ceil(np.sqrt(N)))

    im = np.zeros((10 * l, 10 * l))
    for m in range(l):
        for n in range(l):
            if (m * l + n < N):
                im[10 * m:10 * m + 8, 10 * n:10 * n +
                    8] = x_train[m * l + n].reshape([8, 8])
    plt.imsave('plots/digits.png', im, cmap='Greys')

    n_cluster = 30
    classifier = KMeansClassifier(n_cluster=n_cluster, max_iter=100, e=1e-6)

    classifier.fit(x_train, y_train)
    y_hat_test = classifier.predict(x_test)

    assert y_hat_test.shape == y_test.shape, \
        'y_hat_test and y_test should have same shape'

    print('Prediction accuracy of K-means classifier with {} cluster is {}'.
          format(n_cluster, np.mean(y_hat_test == y_test)))

    linear_classifier = LogisticRegression()
    linear_classifier.fit(x_train, y_train)
    y_hat_test = linear_classifier.predict(x_test)
    print('Accuracy of logistic regression classifier is {}'
          .format(np.mean(y_hat_test == y_test)))

    KNNClassifier = KNeighborsClassifier()
    KNNClassifier.fit(x_train, y_train)
    y_hat_test = KNNClassifier.predict(x_test)
    print('Accuracy of Nearest Neighbour classifier is {}'
          .format(np.mean(y_hat_test == y_test)))

    np.savez('results/k_means_classification.npz',
             y_hat_test=y_hat_test, y_test=y_test, centroids=classifier.centroids, centroid_labels=classifier.centroid_labels)


if __name__ == '__main__':
    kmeans_toy()
    kmeans_image_compression()
    kmeans_classification()
