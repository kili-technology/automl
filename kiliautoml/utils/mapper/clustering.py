import numpy as np
import sklearn
import sklearn.neighbors
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster._agglomerative import _hc_cut, _single_linkage
from sklearn.utils import check_array
from sklearn.utils.validation import check_memory


def aggregate_small_clusters(clustering, limit_size, input_data, method="shared"):
    """Remove the clusters smaller than the limit size
    Reaffect the clusters smaller than the limit size to the closest large cluster
    Args:
        clustering (numpy 1D array): old clustering
        limit_size (int): the smaller size for a cluster
        input_data (numpy 2D array): input data to be used for distance calculation
            between clusters (nb lines : len(clustering))
        method (str): 'shared' to dispatch points on all large clusters, 'new_clust' to
            create a new cluster regrouping all points
    Returns:
        1D numpy_array: new clustering
    """
    new_clustering = clustering.copy()

    for i in range(len(np.unique(new_clustering))):
        new_clustering[new_clustering == np.unique(new_clustering)[i]] = i

    small_clusters = np.unique(new_clustering)[
        np.unique(new_clustering, return_counts=True)[1] < limit_size
    ]

    if len(small_clusters) == 0:
        return new_clustering

    elif len(small_clusters) == len(np.unique(new_clustering)):
        return np.zeros(len(new_clustering)).astype(int)

    else:
        if method == "shared":

            clf = sklearn.neighbors.NearestCentroid()
            clf.fit(input_data, new_clustering)

            distance_between_clusters = sklearn.metrics.pairwise_distances(clf.centroids_)

            distance_between_clusters[:, small_clusters] = np.nan
            closest_large_cluster = np.nanargmin(distance_between_clusters, axis=1)[small_clusters]

            for i in range(len(small_clusters)):
                new_clustering[new_clustering == small_clusters[i]] = closest_large_cluster[i]

        elif method == "new_clust":

            for i in range(len(small_clusters)):
                new_clustering[new_clustering == small_clusters[i]] = np.max(clustering) + 1
        else:
            raise ValueError('method must be "shared" or "new_clust"')

        for i in range(len(np.unique(new_clustering))):
            new_clustering[new_clustering == np.unique(new_clustering)[i]] = i

        return new_clustering


class DensityMergeHierarchicalClustering(ClusterMixin, BaseEstimator):

    """
    DensityMergeHierarchicalClustering.
    Recursively merges pair of clusters of sample data; uses single linkage distance.
    Read more in the orignial paper on Mapper by Singh et al.
    Parameters
    ----------
    n_clusters : int or None, default=None
        The number of clusters to find. It must be ``None`` if
        ``distance_threshold`` or ''n_intervals'' is not ``None``.
    affinity : str or callable, default='euclidean'
        Metric used to compute the linkage. Can be "euclidean", "l1", "l2",
        "manhattan", "cosine", or "precomputed".
        If "precomputed", a distance matrix (instead of a similarity matrix)
        is needed as input for the fit method.
    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.
    connectivity : array-like or callable, default=None
        Connectivity matrix. Defines for each sample the neighboring
        samples following a given structure of the data.
        This can be a connectivity matrix itself or a callable that transforms
        the data into a connectivity matrix, such as derived from
        `kneighbors_graph`. Default is ``None``, i.e, the
        hierarchical clustering algorithm is unstructured.
    distance_threshold : float, default=None
        The linkage distance threshold above which, clusters will not be
        merged. If not ``None``, ``n_clusters`` and ''n_intervals'' must be ``None``
    compute_distances : bool, default=False
        Computes distances between clusters even if `distance_threshold` is not
        used. This can be used to make dendrogram visualization, but introduces
        a computational and memory overhead.
    n_intervals : int, default=30
        number of intervals to split the histogram of distance to reduce
        the number of cluster by 1.
        Used to automatically choose a number of clusters
    n_max_clusters : int, default=10
        maximal number of cluster to create when using the automatic method
        with n_intervals to compute the numbert of cluster automatically
    min_clust_size : int, default= None
        Minimum size of a cluster to be considered
    outliers_agglomeration_method : str, default = 'shared'
        'shared' to dispatch points on all large clusters,
        'new_clust' to create a new cluster regrouping all points
    Attributes
    ----------
    n_clusters_ : int
        The number of clusters found by the algorithm. If
        ``distance_threshold=None``, it will be equal to the given
        ``n_clusters``.
    labels_ : ndarray of shape (n_samples)
        Cluster labels for each point.
    n_leaves_ : int
        Number of leaves in the hierarchical tree.
    n_connected_components_ : int
        The estimated number of connected components in the graph.
    n_features_in_ : int
        Number of features seen during :term:`fit`.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
    children_ : array-like of shape (n_samples-1, 2)
        The children of each non-leaf node. Values less than `n_samples`
        correspond to leaves of the tree which are the original samples.
        A node `i` greater than or equal to `n_samples` is a non-leaf
        node and has children `children_[i - n_samples]`. Alternatively
        at the i-th iteration, children[i][0] and children[i][1]
        are merged to form node `n_samples + i`.
    distances_ : array-like of shape (n_nodes-1,)
        Distances between nodes in the corresponding place in `children_`.
        Only computed if `distance_threshold` or 'n_intervals' is used or `compute_distances`
        is set to `True`.
    """

    def __init__(
        self,
        n_clusters=None,
        *,
        affinity="euclidean",
        memory=None,
        connectivity=None,
        distance_threshold=None,
        compute_distances=False,
        n_intervals=30,
        n_max_clusters=10,
        min_clust_size=None,
        outliers_agglomeration_method="shared"
    ):
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold
        self.memory = memory
        self.connectivity = connectivity
        self.affinity = affinity
        self.compute_distances = compute_distances
        self.n_intervals = n_intervals
        self.n_max_clusters = n_max_clusters
        self.min_clust_size = min_clust_size
        self.outliers_agglomeration_method = outliers_agglomeration_method

    def fit(self, X, y=None):
        """Fit the hierarchical clustering from features, or distance matrix.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) or \
                (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``affinity='precomputed'``.
        y : Ignored
            Not used, present here for API consistency by convention.
        Returns
        -------
        self : object
            Returns the fitted instance.
        """
        X = self._validate_data(X, ensure_min_samples=2)
        return self._fit(X)

    def _fit(self, X):
        """Fit without validation
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``affinity='precomputed'``.
        Returns
        -------
        self : object
            Returns the fitted instance.
        """
        memory = check_memory(self.memory)

        if self.n_clusters is not None and self.n_clusters <= 0:
            raise ValueError(
                "n_clusters should be an integer greater than 0. %s was provided."
                % str(self.n_clusters)
            )

        connectivity = self.connectivity
        if self.connectivity is not None:
            if callable(self.connectivity):
                connectivity = self.connectivity(X)
            connectivity = check_array(connectivity, accept_sparse=["csr", "coo", "lil"])

        n_clusters = None

        if self.min_clust_size is None:
            min_size_clust = max(int(X.shape[0] / 100), 2)
        else:
            min_size_clust = self.min_clust_size

        # Construct the tree
        kwargs = {}
        kwargs["linkage"] = "single"
        kwargs["affinity"] = self.affinity

        return_distance = (
            (self.distance_threshold is not None)
            or (self.n_intervals is not None)
            or self.compute_distances
        )

        out = memory.cache(_single_linkage)(
            X,
            connectivity=connectivity,
            n_clusters=n_clusters,
            return_distance=return_distance,
            **kwargs,
        )
        (self.children_, self.n_connected_components_, self.n_leaves_, parents) = out[:4]

        if return_distance:
            self.distances_ = out[-1]

        # We use the a method to automatically select the number of clusters
        if self.distance_threshold is None and self.n_clusters is None:

            clust_size = np.ones((max(self.children_[-1]) + 1))
            for i in range(len(self.children_) - 1):
                clust_size[self.n_leaves_ + i] = (
                    clust_size[self.children_[i][0]] + clust_size[self.children_[i][1]]
                )

            add_clust_size = np.zeros(len(self.children_))
            for i in range(len(self.children_)):
                add_clust_size[i] = min(
                    clust_size[self.children_[i][0]], clust_size[self.children_[i][1]]
                )

            histogram_distance = np.histogram(
                self.distances_[add_clust_size > min_size_clust], bins=self.n_intervals
            )

            idx = 0
            nb_zeros = len(np.argwhere(histogram_distance[0] == 0))

            if nb_zeros == 0:
                distance_threshold = np.max(self.distances_) + 1

            else:
                n_clusters = np.sum(
                    histogram_distance[0][np.argwhere(histogram_distance[0] == 0)[0][0] :]
                )
                # We enter the loop if the number of cluster created is larger than n_max_clusters
                while n_clusters > self.n_max_clusters and idx < (nb_zeros - 1):
                    idx += 1
                    n_clusters = np.sum(
                        histogram_distance[0][np.argwhere(histogram_distance[0] == 0)[idx][0] :]
                    )

                # if there is no more empty bins in the histogram
                if idx == nb_zeros:
                    distance_threshold = np.max(self.distances_) + 1

                else:
                    distance_threshold = histogram_distance[1][
                        np.argwhere(histogram_distance[0] == 0)[idx][0]
                    ]

            n_clusters = np.count_nonzero(self.distances_ >= distance_threshold) + 1
            # Cut the tree
            self.labels_ = _hc_cut(n_clusters, self.children_, self.n_leaves_)

        if self.distance_threshold is not None:  # distance_threshold is used
            self.n_clusters_ = np.count_nonzero(self.distances_ >= self.distance_threshold) + 1
            # Cut the tree
            self.labels_ = _hc_cut(self.n_clusters_, self.children_, self.n_leaves_)

        elif self.n_clusters is not None:  # n_clusters is used
            self.n_clusters_ = self.n_clusters
            # Cut the tree
            self.labels_ = _hc_cut(self.n_clusters_, self.children_, self.n_leaves_)

        self.labels_ = aggregate_small_clusters(
            self.labels_, min_size_clust, X, self.outliers_agglomeration_method
        )

        return self
