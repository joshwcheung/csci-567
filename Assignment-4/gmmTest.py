import numpy as np
from data_loader import toy_dataset, load_digits
from gmm import GMM
from utils import Figure
from matplotlib.patches import Ellipse


def compute_elipse_params(variance):
    '''
        Compute elipse params for plotting from variance
    '''

    # http://www.cs.cornell.edu/cv/OtherPdf/Ellipse.pdf Slide 17
    # https://stackoverflow.com/a/41821484

    variance_inv = np.linalg.inv(variance)
    a = variance_inv[0, 0]
    c = variance_inv[1, 1]
    b = variance_inv[0, 1] + variance_inv[1, 0]

    M = (variance_inv + variance_inv.T) / 2
    eig, _ = np.linalg.eig(M)
    if (np.abs(eig[0] - a) < np.abs(eig[0] - c)):
        lambda1, lambda2 = eig
    else:
        lambda2, lambda1 = eig

    angle = np.arctan(b / (a - c)) / 2
    return np.sqrt(1 / lambda1), np.sqrt(1 / lambda2), angle


################################################################################
# GMM on 2D toy dataset
# The dataset is generated from N gaussian distributions equally spaced on N radius circle.
# Here, N=4
# You should be able to visualize the learnt gaussian distribution in plots folder
# Complete implementation of fit function for GMM class in gmm.py
################################################################################
x, y = toy_dataset(4, 100)
init = ['k_means', 'random']

for i in init:
    n_cluster = 4
    gmm = GMM(n_cluster=n_cluster, max_iter=1000, init=i, e=1e-6)
    iterations = gmm.fit(x)
    ll = gmm.compute_log_likelihood(x)

    assert gmm.means.shape == (
        n_cluster, 2), 'means should be numpy array with {}X2 shape'.format(n_cluster)

    assert gmm.variances.shape == (
        n_cluster, 2, 2), 'variances should be numpy array with {}X2X2 shape'.format(n_cluster)

    assert gmm.pi_k.shape == (
        n_cluster,), 'pi_k should be numpy vector of size'.format(n_cluster)

    assert iterations > 0 and type(
        iterations) == int, 'Number of updates should be positive integer'

    assert type(ll) == float, 'log-likelihood should be float'

    print('GMM for toy dataset with {} init converged in {} iteration. Final log-likelihood of data: {}'.format(
        i, iterations, ll))

    np.savez('results/gmm_toy_{}.npz'.format(i), iterations=iterations,
             variances=gmm.variances, pi_k=gmm.pi_k, means=gmm.means, log_likelihood=ll, x=x, y=y)

    # plot
    fig = Figure()
    fig.ax.scatter(x[:, 0], x[:, 1], c=y)
    # fig.ax.scatter(gmm.means[:, 0], gmm.means[:, 1], c='red')
    for component in range(n_cluster):
        a, b, angle = compute_elipse_params(gmm.variances[component])
        e = Ellipse(xy=gmm.means[component], width=a * 5, height=b * 5,
                    angle=angle, alpha=gmm.pi_k[component])
        fig.ax.add_artist(e)
    fig.savefig('plots/gmm_toy_dataset_{}.png'.format(i))


################################################################################
# GMM on digits dataset
# We fit a gaussian distribution on digits dataset and show generate samples from the distribution
# Complete implementation of sample function for GMM class in gmm.py
################################################################################

x_train, x_test, y_train, y_test = load_digits()

for i in init:
    n_cluster = 30
    gmm = GMM(n_cluster=n_cluster, max_iter=1000, init=i, e=1e-10)
    iterations = gmm.fit(x_train)
    ll = gmm.compute_log_likelihood(x_train)
    print('GMM for digits dataset with {} init converged in {} iterations. Final log-likelihood of data: {}'.format(i, iterations, ll))

    # plot cluster means
    means = gmm.means
    from matplotlib import pyplot as plt
    l = int(np.ceil(np.sqrt(n_cluster)))

    im = np.zeros((10 * l, 10 * l))
    for m in range(l):
        for n in range(l):
            if (m * l + n < n_cluster):
                im[10 * m:10 * m + 8, 10 * n:10 * n +
                    8] = means[m * l + n].reshape([8, 8])
    im = (im > 0) * im
    plt.imsave('plots/means_{}.png'.format(i), im, cmap='Greys')

    # plot samples
    N = 100
    l = int(np.ceil(np.sqrt(N)))
    samples = gmm.sample(N)

    assert samples.shape == (
        N, x_train.shape[1]), 'Samples should be numpy array with dimensions {}X{}'.format(N, x_train.shape[1])

    im = np.zeros((10 * l, 10 * l))
    for m in range(l):
        for n in range(l):
            if (m * l + n < N):
                im[10 * m: 10 * m + 8, 10 * n: 10 * n +
                    8] = samples[m * l + n].reshape([8, 8])
    im = (im > 0) * im
    plt.imsave('plots/samples_{}.png'.format(i), im, cmap='Greys')

    np.savez('results/gmm_digits_{}.npz'.format(i), iterations=np.array(
        [iterations]), variances=gmm.variances, pi_k=gmm.pi_k, means=gmm.means, samples=samples, log_likelihood=ll, x=x_test, y=y_test)
