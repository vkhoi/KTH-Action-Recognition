import argparse
import numpy as np
import os
import pickle

from sklearn.svm import SVC

CATEGORIES = ["boxing", "handclapping", "handwaving", "jogging", "running", 
    "walking"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SVM classifier")
    parser.add_argument("--svm_file", type=str, default="data/svm_C1_c500.p",
        help="path to svm file")
    parser.add_argument("--bow_file", type=str, default="data/train_bow_c500.p",
        help="path to bow file")

    args = parser.parse_args()
    bow_file = args.bow_file
    svm_file = args.svm_file

    data = pickle.load(open(bow_file, "rb"))

    # Load trained SVM classifier.
    clf = pickle.load(open(svm_file, "rb"))

    confusion_matrix = np.zeros((6, 6))

    correct = 0
    for video in data:

        predicted = clf.predict([video["features"]])

        # Check if majority is correct.
        if predicted == video["category"]:
            correct += 1

        confusion_matrix[CATEGORIES.index(predicted),
            CATEGORIES.index(video["category"])] += 1

    print("%d/%d Correct" % (correct, len(data)))
    print("Accuracy =", correct / len(data))
    
    print("Confusion matrix")
    print(confusion_matrix)
