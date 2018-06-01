import numpy as np
from kmeans import KMeans


class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures
            e : error tolerance
            max_iter : maximum number of updates
            init : initialization of means and variance
                Can be 'random' or 'kmeans'
            means : means of gaussian mixtures
            variances : variance of gaussian mixtures
            pi_k : mixture probabilities of different component
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None

    def fit(self, x):
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if (self.init == 'k_means'):
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k

            # DONOT MODIFY CODE ABOVE THIS LINE
            
            # Run k-means and initialize mu, gamma
            k_means = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e)
            self.means, gamma, _ = k_means.fit(x)
            gamma = np.eye(self.n_cluster)[gamma]
            
            # Initialize N, Sigma, pi
            n = np.sum(gamma, axis=0)
            self.variances = np.zeros((self.n_cluster, D, D))
            for k in range(self.n_cluster):
                x_k = x - self.means[k]
                self.variances[k] = np.dot(np.multiply(x_k.T, gamma[:, k]), x_k) / n[k]
            self.pi_k = n / N
            
            # DONOT MODIFY CODE BELOW THIS LINE

        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - compute variance and pi_k

            # DONOT MODIFY CODE ABOVE THIS LINE
            
            # Initialize N, mu, Sigma, pi
            n = np.zeros(self.n_cluster)
            self.means = np.random.uniform(0, 1, (self.n_cluster, D))
            self.variances = np.array([np.eye(D)] * self.n_cluster)
            self.pi_k = np.full(self.n_cluster, 1 / self.n_cluster)
            
            # Initialize gamma
            gamma = np.zeros((N, self.n_cluster))
            for k in range(self.n_cluster):
                mu = self.means[k]
                sigma = np.copy(self.variances[k])
                rank = np.linalg.matrix_rank(sigma)
                while rank < D:
                    sigma += np.eye(D) * 1e-3
                    rank = np.linalg.matrix_rank(sigma)
                det = np.linalg.det(sigma)
                denom = np.sqrt((2 * np.pi) ** D * det)
                f = np.exp(-0.5 * np.sum(np.multiply(np.dot(x - mu, np.linalg.inv(sigma)), x - mu), axis=1)) / denom
                gamma[:, k] = self.pi_k[k] * f
            gamma = (gamma.T / np.sum(gamma, axis=1)).T
            
            # DONOT MODIFY CODE BELOW THIS LINE

        else:
            raise Exception('Invalid initialization provided')

        # TODO
        # - comment/remove the exception
        # - find the optimal means, variances, and pi_k and assign it to self
        # - return number of updates done to reach the optimal values.
        # Hint: Try to seperate E & M step for clarity

        # DONOT MODIFY CODE ABOVE THIS LINE
        
        l = float('-inf')
        iter = 0
        while iter < self.max_iter:
            iter += 1
            # E step
            for k in range(self.n_cluster):
                mu = self.means[k]
                sigma = np.copy(self.variances[k])
                rank = np.linalg.matrix_rank(sigma)
                while rank < D:
                    sigma += np.eye(D) * 1e-3
                    rank = np.linalg.matrix_rank(sigma)
                det = np.linalg.det(sigma)
                denom = np.sqrt((2 * np.pi) ** D * det)
                f = np.exp(-0.5 * np.sum(np.multiply(np.dot(x - mu, np.linalg.inv(sigma)), x - mu), axis=1)) / denom
                gamma[:, k] = self.pi_k[k] * f
            
            # Compute log-likelihood
            l_new = np.sum(np.log(np.sum(gamma, axis=1)))
            
            # Resume E step
            gamma = (gamma.T / np.sum(gamma, axis=1)).T
            n = np.sum(gamma, axis=0)
            
            # M step
            for k in range(self.n_cluster):
                self.means[k] = np.sum(gamma[:, k] * x.T, axis=1).T / n[k]
                self.variances[k] = np.dot(np.multiply((x - self.means[k]).T, gamma[:, k]), x - self.means[k]) / n[k]
            self.pi_k = n / N
            
            if np.abs(l - l_new) <= self.e:
                break
            l = l_new
        return iter
        
        # DONOT MODIFY CODE BELOW THIS LINE

    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples

        # DONOT MODIFY CODE ABOVE THIS LINE
        
        D = self.means.shape[1]
        
        z = np.random.choice(self.n_cluster, N, p=self.pi_k)
        samples = np.zeros((N, D))
        for i, k in enumerate(z):
            mu = self.means[k]
            sigma = self.variances[k]
            samples[i] = np.random.multivariate_normal(mu, sigma)
        return samples
        
        # DONOT MODIFY CODE BELOW THIS LINE

    def compute_log_likelihood(self, x):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2,  'x can only be 2 dimensional'
        # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood
        # Note: you can call this function in fit function (if required)
        # DONOT MODIFY CODE ABOVE THIS LINE
        
        N, D = x.shape
        l = 0
        joint = np.zeros((N, self.n_cluster))
        for k in range(self.n_cluster):
            mu = self.means[k]
            sigma = np.copy(self.variances[k])
            rank = np.linalg.matrix_rank(sigma)
            while rank < D:
                sigma += np.eye(D) * 1e-3
                rank = np.linalg.matrix_rank(sigma)
            det = np.linalg.det(sigma)
            denom = np.sqrt((2 * np.pi) ** D * det)
            f = np.exp(-0.5 * np.sum(np.multiply(np.dot(x - mu, np.linalg.inv(sigma)), x - mu), axis=1)) / denom
            joint[:, k] = self.pi_k[k] * f
        return float(np.sum(np.log(np.sum(joint, axis=1))))
        
        # DONOT MODIFY CODE BELOW THIS LINE
