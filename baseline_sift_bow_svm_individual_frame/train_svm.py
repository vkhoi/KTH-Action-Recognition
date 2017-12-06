import argparse
import numpy as np
import os
import pickle

from sklearn.svm import SVC

def make_dataset(dataset):
    X = []
    Y = []

    for video in dataset:
        for frame in video["features"]:
            X.append(frame)
            Y.append(video["category"])

    return X, Y

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SVM on bow vectors")
    parser.add_argument("--dataset_bow", type=str, default="data/train_bow_c1000.p",
        help="path to dataset bow file")
    parser.add_argument("--C", type=float, default=1)
    parser.add_argument("--output", type=str, default="data/svm_C1_c1000.p")

    args = parser.parse_args()
    dataset_bow = args.dataset_bow
    C = args.C
    output = args.output

    # Load and make dataset.
    dataset = pickle.load(open(dataset_bow, "rb"))
    X, Y = make_dataset(dataset)

    # Train SVM and save to file.
    clf = SVC(C=C, kernel="linear", verbose=True)
    clf.fit(X, Y)
    pickle.dump(clf, open(output, "wb"))
