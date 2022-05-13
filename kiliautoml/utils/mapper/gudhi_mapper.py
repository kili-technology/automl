import base64
import io
import itertools
import os
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gudhi import SimplexTree
from PIL import Image
from scipy.spatial.distance import directed_hausdorff
from skimage.util import img_as_ubyte
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import pairwise_distances


class _MapperComplex(BaseEstimator, TransformerMixin):
    """
    This is a class for computing Mapper simplicial complexes on point clouds or
        distance matrices. Used internally.
    """

    def __init__(
        self,
        cover_type,
        assignments,
        filters,
        filter_bnds,
        colors,
        resolutions,
        gains,
        inp="point cloud",
        clustering=DBSCAN(),
        mask=0,
        N=100,
        beta=0.0,
        C=10.0,
    ):
        (
            self.filters,
            self.filter_bnds,
            self.resolutions,
            self.gains,
            self.colors,
            self.clustering,
        ) = (filters, filter_bnds, resolutions, gains, colors, clustering)
        self.cover_type, self.assignments = cover_type, assignments
        self.input, self.mask, self.N, self.beta, self.C = inp, mask, N, beta, C

    def estimate_scale(self, X, N=100, inp="point cloud", beta=0.0, C=10.0):
        """
        Compute estimated scale of a point cloud or a distance matrix.
        Parameters:
            X (numpy array of shape (num_points) x (num_coordinates) if point cloud and
                (num_points) x (num_points) if distance matrix):
                input point cloud or distance matrix.
            N (int): subsampling iterations (default 100).
                See http://www.jmlr.org/papers/volume19/17-291/17-291.pdf for details.
            inp (string): either "point cloud" or "distance matrix".
                Type of input data (default "point cloud").
            beta (double): exponent parameter (default 0.).
                See http://www.jmlr.org/papers/volume19/17-291/17-291.pdf for details.
            C (double): constant parameter (default 10.).
                See http://www.jmlr.org/papers/volume19/17-291/17-291.pdf for details.
        Returns:
            delta (double): estimated scale that can be used with eg agglomerative clustering.
        """
        num_pts = X.shape[0]
        delta, m = 0.0, int(num_pts / np.exp((1 + beta) * np.log(np.log(num_pts) / np.log(C))))
        for _ in range(N):
            subpop = np.random.choice(num_pts, size=m, replace=False)
            if inp == "point cloud":
                d, _, _ = directed_hausdorff(X, X[subpop, :])
            if inp == "distance matrix":
                d = np.max(np.min(X[:, subpop], axis=1), axis=0)
            delta += d / N
        return delta

    def get_optimal_parameters_for_agglomerative_clustering(self, X, beta=0.0, C=10.0, N=100):
        """
        Compute optimal scale and resolutions for a point cloud or a distance matrix.
        Parameters:
            X (numpy array of shape (num_points) x (num_coordinates) if point cloud and
                (num_points) x (num_points) if distance matrix):
                input point cloud or distance matrix.
            beta (double): exponent parameter (default 0.).
                See http://www.jmlr.org/papers/volume19/17-291/17-291.pdf for details.
            C (double): constant parameter (default 10.).
                See http://www.jmlr.org/papers/volume19/17-291/17-291.pdf for details.
            N (int): subsampling iterations (default 100).
                See http://www.jmlr.org/papers/volume19/17-291/17-291.pdf for details.
        Returns:
            delta (double): optimal scale that can be used with agglomerative clustering.
            resolutions (numpy array of shape (num_filters):
                optimal resolutions associated to each filter.
        """
        num_filt, delta = self.filters.shape[1], 0
        delta = self.estimate_scale(X=X, N=N, inp=self.input, C=C, beta=beta)

        pairwise = pairwise_distances(X, metric="euclidean") if self.input == "point cloud" else X
        pairs = np.argwhere(pairwise <= delta)
        num_pairs = pairs.shape[0]
        res = []
        for f in range(num_filt):
            F = self.filters[:, f]
            minf, maxf = np.min(F), np.max(F)
            resf = 0
            for p in range(num_pairs):
                resf = max(resf, abs(F[pairs[p, 0]] - F[pairs[p, 1]]))
            res.append(int((maxf - minf) / resf))

        return delta, np.array(res)

    def fit(self, X, y=None):

        # Initialize attributes
        self.mapper_, self.node_info_ = SimplexTree(), {}

        if self.cover_type == "precomputed":

            num_pts = len(self.assignments)

            # Build the binned_data map that takes a patch or a patch intersection and
            # outputs the indices of the points contained in it
            binned_data = {}
            for i, assigment in enumerate(self.assignments):
                for pre_idx in assigment:
                    try:
                        binned_data[pre_idx].append(i)
                    except KeyError:
                        binned_data[pre_idx] = [i]

        else:

            num_pts, num_filters = self.filters.shape[0], self.filters.shape[1]

            # If some resolutions are not specified, automatically compute them
            if self.resolutions is None or self.clustering is None:
                delta, resolutions = self.get_optimal_parameters_for_agglomerative_clustering(
                    X=X, beta=self.beta, C=self.C, N=self.N
                )
                if self.clustering is None:
                    if self.input == "point cloud":
                        self.clustering = AgglomerativeClustering(
                            n_clusters=None,
                            linkage="single",
                            distance_threshold=delta,
                            affinity="euclidean",
                        )
                    else:
                        self.clustering = AgglomerativeClustering(
                            n_clusters=None,
                            linkage="single",
                            distance_threshold=delta,
                            affinity="precomputed",
                        )
                if self.resolutions is None:
                    self.resolutions = resolutions
                    self.resolutions = np.array([int(r) for r in self.resolutions])

            if self.gains is None:
                self.gains = 0.33 * np.ones(num_filters)

            # If some filter limits are unspecified, automatically compute them
            if self.filter_bnds is None:
                self.filter_bnds = np.hstack(
                    [
                        np.min(self.filters, axis=0)[:, np.newaxis],
                        np.max(self.filters, axis=0)[:, np.newaxis],
                    ]
                )

            if np.all(self.gains < 0.5):

                # Compute which points fall in which patch or patch intersections
                interval_inds, intersec_inds = np.empty(self.filters.shape), np.empty(
                    self.filters.shape
                )
                for i in range(num_filters):
                    f, r, g = self.filters[:, i], self.resolutions[i], self.gains[i]
                    min_f, max_f = self.filter_bnds[i, 0], np.nextafter(
                        self.filter_bnds[i, 1], np.inf
                    )
                    interval_endpoints, step = np.linspace(min_f, max_f, num=r + 1, retstep=True)
                    intersec_endpoints = []
                    for j in range(1, len(interval_endpoints) - 1):
                        intersec_endpoints.append(interval_endpoints[j] - g * step / (2 - 2 * g))
                        intersec_endpoints.append(interval_endpoints[j] + g * step / (2 - 2 * g))
                    interval_inds[:, i] = np.digitize(f, interval_endpoints)
                    intersec_inds[:, i] = 0.5 * (np.digitize(f, intersec_endpoints) + 1)

                # Build the binned_data map that takes a patch or a patch intersection
                # and outputs the indices of the points contained in it
                binned_data = {}
                for i in range(num_pts):
                    list_preimage = []
                    for j in range(num_filters):
                        a, b = interval_inds[i, j], intersec_inds[i, j]
                        list_preimage.append([a])
                        if b == a:
                            list_preimage[j].append(a + 1)
                        if b == a - 1:
                            list_preimage[j].append(a - 1)
                    list_preimage = list(itertools.product(*list_preimage))
                    for pre_idx in list_preimage:
                        try:
                            binned_data[pre_idx].append(i)
                        except KeyError:
                            binned_data[pre_idx] = [i]

            else:

                # Compute interval endpoints for each filter
                l_int, r_int = [], []
                for i in range(num_filters):
                    L, R = [], []
                    f, r, g = self.filters[:, i], self.resolutions[i], self.gains[i]
                    min_f, max_f = self.filter_bnds[i, 0], np.nextafter(
                        self.filter_bnds[i, 1], np.inf
                    )
                    interval_endpoints, lstep = np.linspace(min_f, max_f, num=r + 1, retstep=True)
                    for j in range(len(interval_endpoints) - 1):
                        L.append(interval_endpoints[j] - g * step / (2 - 2 * g))
                        R.append(interval_endpoints[j + 1] + g * step / (2 - 2 * g))
                    l_int.append(L)
                    r_int.append(R)

                # Build the binned_data map that takes a patch or a patch intersection and
                # outputs the indices of the points contained in it
                binned_data = {}
                for i in range(num_pts):
                    list_preimage = []
                    for j in range(num_filters):
                        fval = self.filters[i, j]
                        start, end = int(min(np.argwhere(np.array(r_int[j]) >= fval))), int(
                            max(np.argwhere(np.array(l_int[j]) <= fval))
                        )
                        list_preimage.append(list(range(start, end + 1)))
                    list_preimage = list(itertools.product(*list_preimage))
                    for pre_idx in list_preimage:
                        try:
                            binned_data[pre_idx].append(i)
                        except KeyError:
                            binned_data[pre_idx] = [i]

        # Initialize the cover map, that takes a point and outputs the clusters to which it belongs
        cover, clus_base = [[] for _ in range(num_pts)], 0

        # For each patch
        for preimage in binned_data:

            # Apply clustering on the corresponding subpopulation
            idxs = np.array(binned_data[preimage])
            if len(idxs) > 1:
                clusters = (
                    self.clustering.fit_predict(X[idxs, :])
                    if self.input == "point cloud"
                    else self.clustering.fit_predict(X[idxs, :][:, idxs])
                )
            elif len(idxs) == 1:
                clusters = np.array([0])
            else:
                continue

            # Collect various information on each cluster
            num_clus_pre = np.max(clusters) + 1
            for clus_i in range(num_clus_pre):
                node_name = clus_base + clus_i
                subpopulation = idxs[clusters == clus_i]
                self.node_info_[node_name] = {}
                self.node_info_[node_name]["indices"] = subpopulation
                self.node_info_[node_name]["size"] = len(subpopulation)
                self.node_info_[node_name]["colors"] = np.mean(
                    self.colors[subpopulation, :], axis=0
                )
                self.node_info_[node_name]["patch"] = preimage

            # Update the cover map
            for pt in range(clusters.shape[0]):
                node_name = clus_base + clusters[pt]
                if clusters[pt] != -1 and self.node_info_[node_name]["size"] >= self.mask:
                    cover[idxs[pt]].append(node_name)

            clus_base += np.max(clusters) + 1

        # Insert the simplices of the Mapper complex
        for i in range(num_pts):
            self.mapper_.insert(cover[i], filtration=-3)

        return self


class CoverComplex(BaseEstimator, TransformerMixin):
    """
    This class wraps Mapper, Nerve and Graph Induced complexes in a single interface.
    Graph Induced and Nerve complexes can still be called from the class NGIComplex
    (with a few more functionalities, such as defining datasets or graphs through files).
    Key differences between Mapper, Nerve and Graph Induced complexes (GIC) are: Mapper
    nodes are defined with given input clustering method while GIC nodes are defined
    with given input graph and Nerve nodes are defined with cover elements, GIC accepts
    partitions instead of covers while Mapper and Nerve require cover elements to overlap.
    Also, note that when the cover is functional (i.e., preimages of filter functions),
    GIC only accepts one scalar-valued filter with gain < 0.5, meaning that the arguments
    "resolutions" and "gains" should have length 1. If you have more than one scalar filter,
    or if the gain is more than 0.5, the cover should be computed beforehand and fed to the
    class with the "assignments" argument. On the other hand, Mapper and Nerve complexes
    accept "resolutions" and "gains" with any length.
    Attributes:
        simplex_tree (gudhi SimplexTree): simplicial complex representing the
            cover complex computed after calling the fit() method.
        node_info_ (dictionary): various information associated to the nodes of the cover complex.
    """

    def __init__(
        self,
        complex_type="mapper",
        input_type="point cloud",
        cover="functional",
        colors=None,
        mask=0,
        voronoi_samples=100,
        assignments=None,
        filters=None,
        filter_bnds=None,
        resolutions=None,
        gains=None,
        N=100,
        beta=0.0,
        C=10.0,
        clustering=None,
        graph="rips",
        rips_threshold=None,
        distance_matrix_name="",
        input_name="data",
        cover_name="cover",
        color_name="color",
        verbose=False,
    ):
        """
        Constructor for the CoverComplex class.
        Parameters:
            complex_type (string): Only "mapper"
            input_type (string): type of input data. Either "point cloud" or "distance matrix".
            cover (string): specifies the cover.
                Either "functional" (preimages of filter function), "voronoi" or "precomputed".
            colors (numpy array of shape (num_points) x (num_colors)): functions used to color
                the nodes of the cover complex. More specifically, coloring is done by
                computing the means of these functions on the subpopulations corresponding
                to each node. If None, first coordinate is used if input is point cloud,
                and eccentricity is used if input is distance matrix.
            mask (int): threshold on the size of the cover complex nodes (default 0).
                Any node associated to a subpopulation with less than
                **mask** points will be removed.
            voronoi_samples (int): number of Voronoi germs used for partitioning the input dataset.
                Used only if complex_type = "gic" and cover = "voronoi".
            assignments (list of length (num_points) of lists of integers): cover assignment
                for each point. Used only if complex_type = "gic" or "nerve"
                and cover = "precomputed".
            filters (numpy array of shape (num_points) x (num_filters)):
                filter functions (sometimes called lenses)
                used to compute the cover. Each column of the numpy array defines
                a scalar function defined on the input points.
                Used only if cover = "functional".
            filter_bnds (numpy array of shape (num_filters) x 2): limits of each filter,
                of the form [[f_1^min, f_1^max],
                ..., [f_n^min, f_n^max]]. If one of the values is numpy.nan, it can be computed from
                the dataset with the fit() method. Used only if cover = "functional".
            resolutions (numpy array of shape num_filters containing integers): resolution of each
                filter function, ie number of intervals required to cover each filter image.
                Must be of length 1 if complex_type = "gic". Used only if cover = "functional".
                If None, it is estimated from data.
            gains (numpy array of shape num_filters containing doubles in [0,1]):
                gain of each filter function,
                ie overlap percentage of the intervals covering each filter image.
                Must be of length 1
                if complex_type = "gic". Used only if cover = "functional".
            N (int): subsampling iterations (default 100) for estimating scale and resolutions.
                Used only if cover = "functional" and clustering or resolutions = None.
                See http://www.jmlr.org/papers/volume19/17-291/17-291.pdf for details.
            beta (double): exponent parameter (default 0.) for estimating scale and resolutions.
                Used only if cover = "functional" and clustering or resolutions = None.
                See http://www.jmlr.org/papers/volume19/17-291/17-291.pdf for details.
            C (double): constant parameter (default 10.) for estimating scale and resolutions.
                Used only if cover = "functional" and clustering or resolutions = None.
                See http://www.jmlr.org/papers/volume19/17-291/17-291.pdf for details.
            clustering (class): clustering class (default sklearn.cluster.DBSCAN()).
                Common clustering classes can be found in the scikit-learn library
                (such as AgglomerativeClustering for instance).
                Used only if complex_type = "mapper".
                If None, it is set to hierarchical clustering, with scale estimated from data.
            graph (string): type of graph to use for GIC. Used only if complex_type = "gic".
                Currently accepts "rips" only.
            rips_threshold (float): Rips parameter. Used only if complex_type = "gic"
                and graph = "rips".
            distance_matrix_name (string): name of distance matrix. Used when generating plots.
            input_name (string): name of dataset. Used when generating plots.
            cover_name (string): name of cover. Used when generating plots.
            color_name (string): name of color function. Used when generating plots.
            verbose (bool): whether to display info while computing.
        """

        self.complex_type, self.input_type, self.cover, self.colors, self.mask = (
            complex_type,
            input_type,
            cover,
            colors,
            mask,
        )
        (
            self.voronoi_samples,
            self.assignments,
            self.filters,
            self.filter_bnds,
            self.resolutions,
            self.gains,
            self.clustering,
        ) = (voronoi_samples, assignments, filters, filter_bnds, resolutions, gains, clustering)
        self.graph, self.rips_threshold, self.N, self.beta, self.C = (
            graph,
            rips_threshold,
            N,
            beta,
            C,
        )
        (
            self.distance_matrix_name,
            self.input_name,
            self.cover_name,
            self.color_name,
            self.verbose,
        ) = (distance_matrix_name, input_name, cover_name, color_name, verbose)

    def fit(self, X, y=None):
        """
        Fit the CoverComplex class on a point cloud or a distance matrix: compute the cover complex
        and store it in a simplex tree called simplex_tree.
        Parameters:
            X (numpy array of shape (num_points) x (num_coordinates) if point cloud and
            (num_points) x (num_points) if distance matrix): input point cloud or distance matrix.
            y (n x 1 array): point labels (unused).
        """
        self.data = X

        if self.colors is None:
            if self.input_type == "point cloud":
                self.colors = X[:, 0] if self.complex_type == "gic" else X[:, 0:1]
            elif self.input_type == "distance matrix":
                self.colors = (
                    X.max(axis=0) if self.complex_type == "gic" else X.max(axis=0)[:, np.newaxis]
                )

        if self.filters is None:
            if self.input_type == "point cloud":
                self.filters = X[:, 0] if self.complex_type == "gic" else X[:, 0:1]
            elif self.input_type == "distance matrix":
                self.filters = (
                    X.max(axis=0) if self.complex_type == "gic" else X.max(axis=0)[:, np.newaxis]
                )

        if self.complex_type == "mapper":

            assert self.cover != "voronoi"
            self.complex = _MapperComplex(
                self.cover,
                self.assignments,
                filters=self.filters,
                filter_bnds=self.filter_bnds,
                colors=self.colors,
                resolutions=self.resolutions,
                gains=self.gains,
                inp=self.input_type,
                clustering=self.clustering,
                mask=self.mask,
                N=self.N,
                beta=self.beta,
                C=self.C,
            )
            self.complex.fit(X)
            self.simplex_tree = self.complex.mapper_
            self.node_info = self.complex.node_info_

        return self

    def get_networkx(self, get_attrs=False):
        """
        Turn the 1-skeleton of the cover complex computed after calling
            fit() method into a networkx graph.
        This function requires networkx (https://networkx.org/documentation/stable/install.html).
        Parameters:
            get_attrs (bool): if True, the color functions will be used as attributes
                for the networkx graph.
        Returns:
            G (networkx graph): graph representing the 1-skeleton of the cover complex.
        """
        try:
            import networkx as nx

            st = self.simplex_tree
            G = nx.Graph()
            for splx, _ in st.get_skeleton(1):
                if len(splx) == 1:
                    G.add_node(splx[0])
                if len(splx) == 2:
                    G.add_edge(splx[0], splx[1])
            if get_attrs:
                attrs = {k: {"attr_name": self.node_info[k]["colors"]} for k in G.nodes()}
                nx.set_node_attributes(G, attrs)
            return G
        except ImportError:
            print("Networkx not found, nx graph not computed")

    class _constant_clustering:
        def fit_predict(X):
            return np.zeros([len(X)], dtype=np.int32)

    def print_to_dot(self, epsv=0.2, epss=0.4):
        """
        Write the cover complex in a DOT file, that can be processed with, e.g., neato.
        Parameters:
            epsv (float): scale the node colors between [epsv, 1-epsv]
            epss (float): scale the node sizes between [epss, 1-epss]
        """
        st = self.simplex_tree
        node_info = self.node_info

        maxv, minv = max([node_info[k]["colors"][0] for k in node_info.keys()]), min(
            [node_info[k]["colors"][0] for k in node_info.keys()]
        )
        maxs, mins = max([node_info[k]["size"] for k in node_info.keys()]), min(
            [node_info[k]["size"] for k in node_info.keys()]
        )

        f = open(self.input_name + ".dot", "w")
        f.write("graph MAP{")
        cols = []
        for simplex, _ in st.get_skeleton(0):
            cnode = (
                (1.0 - 2 * epsv) * (node_info[simplex[0]]["colors"][0] - minv) / (maxv - minv)
                + epsv
                if maxv != minv
                else 0
            )
            snode = (
                (1.0 - 2 * epss) * (node_info[simplex[0]]["size"] - mins) / (maxs - mins) + epss
                if maxs != mins
                else 1
            )
            f.write(
                str(simplex[0])
                + "[shape=circle width="
                + str(snode)
                + ' fontcolor=black color=black label="'
                + '" style=filled fillcolor="'
                + str(cnode)
                + ', 1, 1"]'
            )
            cols.append(cnode)
        for simplex, _ in st.get_filtration():
            if len(simplex) == 2:
                f.write("  " + str(simplex[0]) + " -- " + str(simplex[1]) + " [weight=15];")
        f.write("}")
        f.close()

        L = np.linspace(epsv, 1.0 - epsv, 100)
        colsrgb = []
        try:
            import colorsys

            for c in L:
                colsrgb.append(colorsys.hsv_to_rgb(c, 1, 1))
            fig, ax = plt.subplots(figsize=(6, 1))
            fig.subplots_adjust(bottom=0.5)
            my_cmap = matplotlib.colors.ListedColormap(colsrgb, name=self.color_name)
            cb = matplotlib.colorbar.ColorbarBase(
                ax,
                cmap=my_cmap,
                norm=matplotlib.colors.Normalize(vmin=minv, vmax=maxv),
                orientation="horizontal",
            )
            cb.set_label(self.color_name)
            fig.savefig("colorbar_" + self.color_name + ".pdf", format="pdf")
            plt.close()
        except ImportError:
            print("colorsys not found, colorbar not printed")

    def print_to_txt(self):
        """
        Write the cover complex to a TXT file, that can be processed with KeplerMapper.
        """
        st = self.simplex_tree
        if self.complex_type == "gic":
            self.complex.write_info(
                self.input_name.encode("utf-8"),
                self.cover_name.encode("utf-8"),
                self.color_name.encode("utf-8"),
            )
        elif self.complex_type == "mapper":
            f = open(self.input_name + ".txt", "w")
            f.write(self.input_name + "\n")
            f.write(self.cover_name + "\n")
            f.write(self.color_name + "\n")
            f.write(str(self.complex.resolutions[0]) + " " + str(self.complex.gains[0]) + "\n")
            f.write(
                str(st.num_vertices())
                + " "
                + str(len(list(st.get_skeleton(1))) - st.num_vertices())
                + "\n"
            )
            name2id = {}
            idv = 0
            for s, _ in st.get_skeleton(0):
                f.write(
                    str(idv)
                    + " "
                    + str(self.node_info[s[0]]["colors"][0])
                    + " "
                    + str(self.node_info[s[0]]["size"])
                    + "\n"
                )
                name2id[s[0]] = idv
                idv += 1
            for s, _ in st.get_skeleton(1):
                if len(s) == 2:
                    f.write(str(name2id[s[0]]) + " " + str(name2id[s[1]]) + "\n")
            f.close()


def extract_node_value_mapper(mapper_cover_complex, interest_values, method="mean"):
    """Extract values of interest per node
    Compute a numpy array with values of interest_values per node in mapper_cover_complex
    Args:
        mapper_cover_complex (dict): Cover complex computed with gudhi_mapper
        interest_values (2D numpy_array - nb of lines = nb_lines of mapper_cover_complex):
            values of interest for the computation of the distance matrix
        method (str among 'mean', 'min', 'max'): the method used to commpute the color of the node
    Returns:
        1D numpy_array: values of interest_values per node in mapper_cover_complex
    """

    graph = mapper_cover_complex.get_networkx()
    nb_nodes = len(graph.nodes)
    nodes = list(graph.nodes)

    node_value_mapper = np.zeros(nb_nodes)

    for i in range(nb_nodes):
        if method == "mean":
            node_value_mapper[i] = np.mean(
                interest_values[mapper_cover_complex.node_info[nodes[i]]["indices"]], axis=0
            )
        elif method == "min":
            node_value_mapper[i] = np.min(
                interest_values[mapper_cover_complex.node_info[nodes[i]]["indices"]], axis=0
            )
        elif method == "max":
            node_value_mapper[i] = np.max(
                interest_values[mapper_cover_complex.node_info[nodes[i]]["indices"]], axis=0
            )
        else:
            raise ValueError("Clustering method not recognized.")

    return node_value_mapper


def data_index_in_mapper(mapper_cover_complex):
    """Extract all data index used in mapper_cover_complex
    Compute a numpy array with all index of data input used in
    kmapper_cover_complex (each index appears as many time as in kmapper)
    Args:
        mapper_cover_complex (dict): Cover complex computed with gudhi_mapper
    Returns:
        1D numpy_array: all data index used in kmapper_cover_complex
    """
    idx_points_in_mapper = np.empty(0)

    for key in mapper_cover_complex.node_info.keys():
        idx_points_in_mapper = np.concatenate(
            (idx_points_in_mapper, mapper_cover_complex.node_info[key]["indices"])
        )

    return idx_points_in_mapper


def display_pic_from_mapper_node(
    mapper_cover_complex,
    node_id,
    nb_max_images=10,
    pict_data_type="pandas_df",
    pict_size=None,
    pict_dataset=None,
    pict_folder=None,
    pict_file_names=None,
):
    """Display a set of picture included in the chosen mapper node
    Choose randomly a maximum of nb_max_images of indices part of selected mapper node.
    Display the related picture, wether the picture are stored in a dataframe or in a picture folder
    Args:
        mapper_cover_complex (dict): Cover complex computed with gudhi_mapper
        node_id (int): node_id from networks or html display
        nb_max_images (int): maximum number of images to display
        pict_data_type (str): either "pandas_df" or "img_link"
        pict_size (tuple): picture size (only required if pict_data_type == "pandas_df")
        pict_dataset (pandas dataframe): dataframe where picture data are stored
            (only required if pict_data_type == "pandas_df")
        pict_folder (str): folder where picture are stored
            (only required if pict_data_type == "img_link")
        pict_file_names (pandas dataframe): dataframe column where picture
            file names are stored (only required if pict_data_type == "img_link")
    Returns:
        nothing, plot pictures
    """
    # Map node id from html picture to mapper's node name
    id2name = {}
    idv = 0
    for s, _ in mapper_cover_complex.simplex_tree.get_skeleton(0):
        id2name[idv] = s[0]
        idv += 1

    if pict_data_type == "pandas_df":
        if not (isinstance(pict_dataset, pd.DataFrame)):
            raise ValueError("pict_dataset must be a pandas dataframe")
        if pict_size is None:
            pict_size = [
                int((pict_dataset.shape[1]) ** (1 / 2)),
                int((pict_dataset.shape[1]) ** (1 / 2)),
            ]
        # Initialize picture
        nb_display = min(
            nb_max_images, len(mapper_cover_complex.node_info[id2name[node_id]]["indices"])
        )
        testImage = np.zeros((pict_size[0], pict_size[1] * nb_display))

        testImage = np.concatenate(
            (
                pict_dataset.to_numpy()[
                    [
                        np.random.choice(
                            mapper_cover_complex.node_info[id2name[node_id]]["indices"],
                            nb_display,
                            replace=False,
                        )
                    ],
                    :,
                ]
            ).reshape(nb_max_images, pict_size[0], pict_size[1]),
            axis=1,
        )

        plt.figure(figsize=(nb_display * 3, 2))
        plt.imshow(testImage)
        plt.show()

    if pict_data_type == "img_link":
        nb_display = min(
            nb_max_images, len(mapper_cover_complex.node_info[id2name[node_id]]["indices"])
        )
        f, axarr = plt.subplots(1, nb_display, figsize=(nb_display * 3, 2))

        if not (isinstance(pict_folder, str)) or not (isinstance(pict_file_names, pd.Series)):
            raise ValueError("pict_folder must be a path and pict_file_names a pandas series")

        list_pict = pict_file_names.to_numpy()[
            np.random.choice(
                mapper_cover_complex.node_info[id2name[node_id]]["indices"],
                nb_display,
                replace=False,
            )
        ]
        for i in range(nb_display):
            axarr[i].imshow(Image.open(os.path.join(pict_folder, list_pict[i])))
        plt.show()


def gudhi_to_KM(mapper_cover_complex):
    """Convert mapper_cover_complex from Gudhi_mapper to a Kepler Mapper ready to use for visualization
    Args:
        mapper_cover_complex (dict): Cover complex computed with gudhi_mapper
    Returns:
        dict ('links', 'meta_data', and 'nodes'): ready to use for visualization
    """
    out = dict()

    # extract metadata
    out["meta_data"] = {
        "clustering": mapper_cover_complex.get_params()["clustering"],
        "resolutions": mapper_cover_complex.get_params()["resolutions"],
        "gains": mapper_cover_complex.get_params()["gains"],
        "cover_name": mapper_cover_complex.get_params()["cover_name"],
        "input_name": mapper_cover_complex.get_params()["input_name"],
    }

    # extract edges / links
    out["links"] = defaultdict(list)
    for s, _ in mapper_cover_complex.simplex_tree.get_skeleton(1):
        if len(s) == 2:
            out["links"][str(s[0])].append(str(s[1]))

    # extract nodes
    out["nodes"] = defaultdict(list)
    for node in mapper_cover_complex.node_info:
        out["nodes"][str(node)] = list(mapper_cover_complex.node_info[node]["indices"])

    return out


def custom_tooltip_picture(
    label,
    pict_data_type="pandas_df",
    pict_dataset=None,
    pict_size=None,
    pict_folder=None,
    pict_file_names=None,
    image_list=None,
):
    """Create numpy array with picture to be used as custom_tooltips in KepplerMapper.visualize
    Args:
        label (numpy array): label to be display with picture in visualization
        pict_data_type (str): either "pandas_df", "img_link", or "img_list"
        pict_size (tuple): picture size (only required if pict_data_type == "pandas_df")
        pict_dataset (pandas dataframe): dataframe where picture data are stored
            (only required if pict_data_type == "pandas_df")
        pict_folder (str): folder where picture are stored
            (only required if pict_data_type == "img_link")
        pict_file_names (pandas dataframe): dataframe column where picture file names are stored
            (only required if pict_data_type == "img_link")
        image_list (list of DownloadedImages): list of downloaded image from a Kili dataset
            (only required if pict_data_type == "img_list")
    Returns:
        Numpy array: ready to use for visualization
    """
    tooltip_s = []

    if pict_data_type == "pandas_df":
        if not (isinstance(pict_dataset, pd.DataFrame)):
            raise ValueError("pict_dataset must be a pandas dataframe")
        if pict_size is None:
            pict_size = [
                int((pict_dataset.shape[1]) ** (1 / 2)),
                int((pict_dataset.shape[1]) ** (1 / 2)),
            ]

        for ys, image_data in zip(label, pict_dataset.to_numpy()):
            output = io.BytesIO()
            # Data was a flat row of "pixels".
            _image_data = img_as_ubyte(image_data.reshape(pict_size))
            img = Image.fromarray(_image_data, "L").resize((64, 64))
            img.save(output, format="PNG")
            contents = output.getvalue()
            img_encoded = base64.b64encode(contents)
            img_tag = """<div style="width:71px;
                                    height:71px;
                                    overflow:hidden;
                                    float:left;
                                    position: relative;">
                        <img src="data:image/png;base64,%s"
                        style="position:absolute; top:0; right:0" />
                        <div style="position: relative; top: 0; left: 1px; font-size:9px">%s</div>
                        </div>""" % (
                (img_encoded.decode("utf-8"), ys)
            )

            tooltip_s.append(img_tag)
            output.close()

    if pict_data_type == "img_link":
        if not (isinstance(pict_folder, str)) or not (isinstance(pict_file_names, pd.Series)):
            raise ValueError("pict_folder must be a path and pict_file_names a pandas series")
        for ys, image_data in zip(label, pict_file_names.to_numpy()):
            output = io.BytesIO()
            # Data was a flat row of "pixels".
            img = Image.open(os.path.join(pict_folder, image_data)).resize((64, 64))
            img.save(output, format="PNG")
            contents = output.getvalue()
            img_encoded = base64.b64encode(contents)
            img_tag = """<div style="width:71px;
                                    height:71px;
                                    overflow:hidden;
                                    float:left;
                                    position: relative;">
                        <img src="data:image/png;base64,%s"
                        style="position:absolute; top:0; right:0" />
                        <div style="position: relative; top: 0; left: 1px; font-size:9px">%s</div>
                        </div>""" % (
                (img_encoded.decode("utf-8"), ys)
            )
            tooltip_s.append(img_tag)
            output.close()

    if pict_data_type == "img_list":
        if not (isinstance(image_list, list)):
            raise ValueError("image_list must be a plist")
        for ys, im in zip(label, image_list):
            output = io.BytesIO()
            # Data was a flat row of "pixels".
            img = im.image.resize((64, 64))
            img.save(output, format="PNG")
            contents = output.getvalue()
            img_encoded = base64.b64encode(contents)
            img_tag = """<div style="width:71px;
                                    height:71px;
                                    overflow:hidden;
                                    float:left;
                                    position: relative;">
                        <img src="data:image/png;base64,%s"
                        style="position:absolute; top:0; right:0" />
                        <div style="position: relative; top: 0; left: 1px; font-size:9px">%s</div>
                        </div>""" % (
                (img_encoded.decode("utf-8"), ys)
            )
            tooltip_s.append(img_tag)
            output.close()

    tooltip_s = np.array(tooltip_s)

    return tooltip_s


def confusion_filter(
    predictions,
    labels=None,
    cover_projection=np.array([[0, 0.5, 0.6, 0.7, 0.8], [0.55, 0.65, 0.75, 0.85, 1]]),
    cover_alt_projection=np.array([[0, 0.15, 0.25, 0.35], [0.2, 0.3, 0.4, 0.5]]),
):
    """Create a custom filter for Mapper according to the confidence of the
        predictions and the true labels (if available)
    Args:
        predictions (2D numpy array): prediction of the ML model.
            Shape = (nb_of_assets, nb_of_classes)
        labels (numpy array): true label of each assets
        cover_projection (numpy array with 2 lines): cover to be used for the
            projected class (each column is an interval for the cover)
        cover_alt_projection (numpy array with 2 lines): cover to be used for the
            alternate projected class (each column is an interval for the cover)
    Returns:
        list of list: each elements of the list is the list of cover element the asset belongs to
    """

    (n_assets, n_classes) = predictions.shape

    # labels are not known for every assets
    if (labels is None) or (len(labels) != n_assets):

        predictions_order = np.argsort(predictions, axis=1)
        assignments = []
        for i in range(n_assets):
            assignment = []

            # we use the first 2 predicted classes to construct the filter
            pred_class = predictions_order[i, n_classes - 1]
            confidence_pred_class = predictions[i, pred_class]
            alt_pred_class = predictions_order[i, n_classes - 2]
            confidence_alt_pred_class = predictions[i, alt_pred_class]

            # We look to which cover the asset belongs to

            # If no prediction is strongly dominent and their is no alternate class significant
            # This is the central nodes of our "Neuron"
            if (confidence_pred_class < cover_projection[1, 0]) and (
                confidence_alt_pred_class < cover_alt_projection[1, 0]
            ):
                assignment.append(
                    pred_class
                    * (cover_projection.shape[1] + n_classes * cover_alt_projection.shape[1])
                )

            # If one prediction is strongly dominent
            # This is the axone of the Neuron
            for p in range(1, cover_projection.shape[1]):
                if cover_projection[0, p] < confidence_pred_class < cover_projection[1, p]:
                    assignment.append(
                        pred_class
                        * (cover_projection.shape[1] + n_classes * cover_alt_projection.shape[1])
                        + p
                    )

            # If no prediction is strongly dominent but their is an significant alternate class
            # This is the central nodes of our "Neuron"
            if confidence_pred_class < cover_projection[0, 1]:
                for a in range(1, cover_alt_projection.shape[1]):
                    if (
                        cover_alt_projection[0, a]
                        < confidence_alt_pred_class
                        < cover_alt_projection[1, a]
                    ):
                        assignment.append(
                            pred_class
                            * (
                                cover_projection.shape[1]
                                + n_classes * cover_alt_projection.shape[1]
                            )
                            + cover_projection.shape[1]
                            + (alt_pred_class * cover_alt_projection.shape[1])
                            + a
                        )

            assignments.append(assignment)

    # if labels are known for every assets
    else:

        assignments = []
        for i in range(n_assets):
            assignment = []

            # we use the label and the predicted class to construct the filter
            label = labels[i]
            pred_class = np.argmax(predictions[i, :])
            confidence_pred_class = predictions[i, pred_class]

            for p in range(cover_projection.shape[1]):
                if cover_projection[0, p] < confidence_pred_class < cover_projection[1, p]:
                    assignment.append(
                        label * n_classes * cover_projection.shape[1]
                        # We only separate asset if the confidence in projected class is significant
                        + (p != 0) * pred_class * cover_projection.shape[1]
                        + p
                    )

            assignments.append(assignment)

    return assignments
