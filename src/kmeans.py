import numpy as np

class KMeans():
    def __init__(self, n_clusters):
        """
        This class implements the traditional KMeans algorithm with hard assignments:

        https://en.wikipedia.org/wiki/K-means_clustering

        The KMeans algorithm has two steps:

        1. Update assignments
        2. Update the means

        While you only have to implement the fit and predict functions to pass the
        test cases, we recommend that you use an update_assignments function and an
        update_means function internally for the class.

        Use only numpy to implement this algorithm.

        Args:
            n_clusters (int): Number of clusters to cluster the given data into.

        """
        self.n_clusters = n_clusters
        self.means = None

    def fit(self, features):
        """
        Fit KMeans to the given data using `self.n_clusters` number of clusters.
        Features can have greater than 2 dimensions.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            None (saves model - means - internally)
        """
        self.means = features[np.random.randint(features.shape[0], size=self.n_clusters), :]
        newmeans = self.means.copy()
        labels = np.zeros(features.shape[0])
        while True:
            for i in range(features.shape[0]):
                distances = np.zeros(self.means.shape[0])
                for j in range(self.means.shape[0]):
                    distances[j] = np.linalg.norm(features[i] - self.means[j])
                labels[i] = np.argmin(distances)
            for i in range(self.means.shape[0]):
                newmeans[i, :] = np.mean(features[np.where(labels == i), :].squeeze(axis=0), axis=0)
            if np.allclose(self.means, newmeans):
                break
            self.means = newmeans


    def predict(self, features):
        """
        Given features, an np.ndarray of size (n_samples, n_features), predict cluster
        membership labels.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            predictions (np.ndarray): predicted cluster membership for each features,
                of size (n_samples,). Each element of the array is the index of the
                cluster the sample belongs to.
        """
        predictions = np.zeros(features.shape[0])
        for i in range(features.shape[0]):
            distances = np.zeros(self.means.shape[0])
            for j in range(self.means.shape[0]):
                distances[j] = np.linalg.norm(features[i] - self.means[j])
            predictions[i] = np.argmin(distances)
        return predictions