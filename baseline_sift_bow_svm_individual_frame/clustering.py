import argparse
import numpy as np
import os
import pickle

from sklearn.cluster import KMeans
from numpy import size

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run KMeans on training set")
    parser.add_argument("--dataset", type=str, default="data/train_keypoints.p",
                        help="number of clusters")
    parser.add_argument("--clusters", type=int, default=1000,
                        help="number of clusters")

    args = parser.parse_args()
    dataset = args.dataset
    clusters = args.clusters

    print("Loading dataset")
    train_features = pickle.load(open(dataset, "rb"))
    n_features = len(train_features)

    print("Number of feature points to run clustering on: %d" % n_features)

    # Clustering with KMeans.
    print("Running KMeans clustering")
    kmeans = KMeans(init='k-means++', n_clusters=clusters, n_init=10, n_jobs=2,
        verbose=1)
    kmeans.fit(train_features)

    # Save trained kmeans object to file.
    pickle.dump(kmeans, open("data/cb_%dclusters.p" % clusters, "wb"))
