import numpy as np
import os
import pickle

CATEGORIES = ["boxing", "handclapping", "handwaving", "jogging", "running", 
    "walking"]

# Dataset are divided according to the instruction at:
# http://www.nada.kth.se/cvap/actions/00sequences.txt
TRAIN_PEOPLE_ID = [11, 12, 13, 14, 15, 16, 17, 18]
DEV_PEOPLE_ID = [19, 20, 21, 23, 24, 25, 1, 4]
TEST_PEOPLE_ID = [22, 2, 3, 5, 6, 7, 8, 9, 10]

if __name__ == "__main__":

    train = []
    dev = []
    test = []

    # Store keypoints in training set.
    train_keypoints = []

    for category in CATEGORIES:
        print("Processing category %s" % category)
        category_features = pickle.load(open("data/optflow_%s.p" % category, "rb"))

        for video in category_features:
            person_id = int(video["filename"].split("_")[0][6:])

            if person_id in TRAIN_PEOPLE_ID:
                train.append(video)

                for frame in video["features"]:
                    train_keypoints.append(frame)

            elif person_id in DEV_PEOPLE_ID:
                dev.append(video)
            else:
                test.append(video)

    print("Saving train/dev/test set to files")
    pickle.dump(train, open("data/train.p", "wb"))
    pickle.dump(dev, open("data/dev.p", "wb"))
    pickle.dump(test, open("data/test.p", "wb"))

    print("Saving keypoints in training set")
    pickle.dump(train_keypoints, open("data/train_keypoints.p", "wb"))

