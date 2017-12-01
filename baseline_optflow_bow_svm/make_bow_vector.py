import argparse
import numpy as np
import os
import pickle

from scipy.cluster.vq import vq

def make_bow(dataset, clusters, tfidf):
    print("Make bow vector for each frame")

    n_videos = len(dataset)

    bow = np.zeros((n_videos, clusters.shape[0]), dtype=np.float)

    # Make bow vectors for all videos.
    video_index = 0
    for video in dataset:
        visual_word_ids = vq(video["features"], clusters)[0]
        for word_id in visual_word_ids:
            bow[video_index, word_id] += 1
        video_index += 1

    # Check whether to use TF-IDF weighting.
    if tfidf:
        print("Applying TF-IDF weighting")
        freq = np.sum((bow > 0) * 1, axis = 0)
        idf = np.log((n_videos + 1) / (freq + 1))
        bow = bow * idf

    # Replace features in dataset with the bow vector we've computed.
    video_index = 0
    for i in range(len(dataset)):

        dataset[i]["features"] = bow[video_index]
        video_index += 1

        if (i + 1) % 50 == 0:
            print("Processed %d/%d videos" % (i + 1, len(dataset)))

    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make bag of words vector")
    parser.add_argument("--codebook", type=str, default="data/cb_500clusters.p",
        help="path to codebook file")
    parser.add_argument("--tfidf", type=int, default=1,
        help="whether to use tfidf weighting")
    parser.add_argument("--dataset", type=str, default="data/train.p",
        help="path to dataset file")
    parser.add_argument("--output", type=str, default="data/train_bow_c500.p",
        help="path to output bow file")

    args = parser.parse_args()
    codebook_file = args.codebook
    tfidf = args.tfidf
    dataset = args.dataset
    output = args.output

    # Load clusters.
    codebook = pickle.load(open(codebook_file, "rb"))
    clusters = codebook.cluster_centers_

    # Load dataset.
    dataset = pickle.load(open(dataset, "rb"))

    # Make bow vectors.
    dataset_bow = make_bow(dataset, clusters, tfidf)

    # Save.
    pickle.dump(dataset_bow, open(output, "wb"))

