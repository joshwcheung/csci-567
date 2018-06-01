import numpy as np


class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple
                (centroids or means, membership, number_of_updates )
            Note: Number of iterations is the number of time you update means other than initialization
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        N, D = x.shape

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership untill convergence or untill you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        
        def distortion(mu, x, r):
            N = x.shape[0]
            return np.sum([np.sum((x[r == k] - mu[k]) ** 2) for k in range(self.n_cluster)]) / N
        
        # Initialize centroids, membership, distortion
        mu = x[np.random.choice(N, self.n_cluster, replace=False), :]
        r = np.zeros(N, dtype=int)
        J = distortion(mu, x, r)
        # Loop until convergence/max_iter
        iter = 0
        while iter < self.max_iter:
            # Compute membership
            l2 = np.sum(((x - np.expand_dims(mu, axis=1)) ** 2), axis=2)
            r = np.argmin(l2, axis=0)
            # Compute distortion
            J_new = distortion(mu, x, r)
            if np.absolute(J - J_new) <= self.e:
                break
            J = J_new
            # Compute means
            mu_new = np.array([np.mean(x[r == k], axis=0) for k in range(self.n_cluster)])
            index = np.where(np.isnan(mu_new))
            mu_new[index] = mu[index]
            mu = mu_new
            iter += 1
        return (mu, r, iter)
        
        # DONOT CHANGE CODE BELOW THIS LINE


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - N size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering
                self.centroid_labels : labels of each centroid obtained by
                    majority voting
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        
        k_means = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e)
        centroids, membership, _ = k_means.fit(x)
        votes = [{} for k in range(self.n_cluster)]
        for y_i, r_i in zip(y, membership):
            if y_i not in votes[r_i].keys():
                votes[r_i][y_i] = 1
            else:
                votes[r_i][y_i] += 1
        centroid_labels = []
        for votes_k in votes:
            if not votes_k:
                centroid_labels.append(0)
            centroid_labels.append(max(votes_k, key=votes_k.get))
        centroid_labels = np.array(centroid_labels)
        
        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a vector of shape {}'.format(
            self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        
        l2 = np.sum(((x - np.expand_dims(self.centroids, axis=1)) ** 2), axis=2)
        r = np.argmin(l2, axis=0)
        return self.centroid_labels[r]
        
        # DONOT CHANGE CODE BELOW THIS LINE
