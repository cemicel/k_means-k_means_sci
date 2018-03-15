from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
import operator
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_random_data():
    x1 = np.random.uniform(-1, 1, 100)
    y1 = np.random.uniform(-1, 1, 100)
    z1 = np.random.uniform(-1, 1, 100)

    x2 = np.random.uniform(-1, 2, 100)
    y2 = np.random.uniform(1, 2, 100)
    z2 = np.random.uniform(1.2, 4, 100)

    x3 = np.random.uniform(-1, -2, 100)
    y3 = np.random.uniform(1, 3, 100)
    z3 = np.random.uniform(3, 5, 100)

    data = [np.array(i) for i in zip(x1, y1, z1)]
    data1 = [np.array(i) for i in zip(x2, y2, z2)]
    data2 = [np.array(i) for i in zip(x3, y3, z3)]

    data.extend(data1)
    data.extend(data2)

    return (np.array(data))


if __name__ == "__main__":

    X = get_random_data()

    np.random.shuffle(X)

    test_range = 10

    dic = {i: 0 for i in range(1, test_range)}

    for k in range(1, test_range):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)

        labels = kmeans.labels_

        centroids = kmeans.cluster_centers_

        bnd = estimate_bandwidth(X, n_samples=len(X))

        ms = MeanShift(bandwidth=bnd, bin_seeding=True)

        ms.fit(X)

        ms_lb = ms.labels_

        cls = ms.cluster_centers_

        lsb_unc = np.unique(ms_lb)

        numb_cls = len(lsb_unc)

        dic[numb_cls] += 1

    sorted_d = sorted(dic.items(), key=operator.itemgetter(1))

    optimal_k = sorted_d[-1][0]

    print('optimal number of clusters: {}'.format(optimal_k))

    optimal_k_means = KMeans(n_clusters=optimal_k, random_state=0).fit(X)

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[:, :1], X[:, 1:2], X[:, 2:3], c=optimal_k_means.labels_.astype(np.float), marker='o')

    plt.show()
