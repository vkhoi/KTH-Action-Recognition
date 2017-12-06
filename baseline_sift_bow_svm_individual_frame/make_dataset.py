import numpy as np
import os
import pickle

CATEGORIES = ["boxing", "handclapping", "handwaving", "jogging", "running", 
    "walking"]

# Number of keypoints to sample in each category in the training set.
N_SAMPLES_TRAINING = 10000

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

    # Keep track of number of frames with human detected for each category in
    # training set.
    human_detected_count = {}

    # Keep track of number of SIFT keypoints for each category in training set.
    keypoints_count = {}

    for category in CATEGORIES:
        print("Processing category %s" % category)
        category_features = pickle.load(open("data/sift_%s.p" % category, "rb"))

        human_detected_count[category] = 0
        keypoints_count[category] = 0

        for video in category_features:
            person_id = int(video["filename"].split("_")[0][6:])

            if person_id in TRAIN_PEOPLE_ID:
                train.append(video)

                # Number of frames with human.
                human_detected_count[category] += len(video["features"])

                # Number of SIFT features in this video.
                for frame in video["features"]:
                    keypoints_count[category] += frame.shape[0]

            elif person_id in DEV_PEOPLE_ID:
                dev.append(video)
            else:
                test.append(video)

    # Make list of sampled keypoints in training set. This list is used for
    # building codebook and BoW vector.
    for category in CATEGORIES:
        category_features = pickle.load(open("data/sift_%s.p" % category, "rb"))

        # Randomly sample N_SAMPLES_TRAINING keypoints in this category.
        samples = set(np.random.choice(keypoints_count[category],
            N_SAMPLES_TRAINING, replace=False))

        index_keypoint = 0

        for video in category_features:
            person_id = int(video["filename"].split("_")[0][6:])

            if person_id in TRAIN_PEOPLE_ID:
                for frame in video["features"]:
                    for i in range(frame.shape[0]):
                        if index_keypoint in samples:
                            train_keypoints.append(frame[i])
                        index_keypoint += 1

    print("Saving train/dev/test set to files")
    pickle.dump(train, open("data/train.p", "wb"))
    pickle.dump(dev, open("data/dev.p", "wb"))
    pickle.dump(test, open("data/test.p", "wb"))

    print("Saving sampled keypoints in training set")
    pickle.dump(train_keypoints, open("data/train_keypoints.p", "wb"))

    print("Number of humans detected & SIFT keypoints for each category")
    for category in CATEGORIES:
        print("%s: %d %d" % (category, human_detected_count[category],
            keypoints_count[category]))

