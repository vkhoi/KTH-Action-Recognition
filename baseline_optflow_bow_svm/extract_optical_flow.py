import cv2
import numpy as np
import os
import pickle

CATEGORIES = ["boxing", "handclapping", "handwaving", "jogging", "running", 
    "walking"]

if __name__ == "__main__":

    # Create directory to store extracted SIFT features.
    os.makedirs("data", exist_ok=True)

    # Setup parameters for optical flow.
    farneback_params = dict(winsize = 20, iterations=1,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN, levels=1,
        pyr_scale=0.5, poly_n=5, poly_sigma=1.1, flow=None)

    n_processed_files = 0

    for category in CATEGORIES:
        print("Processing category %s" % category)

        # Get all files in current category's folder.
        folder_path = os.path.join("..", "dataset", category)
        filenames = os.listdir(folder_path)

        # List to store features. features[i] stores features for the i-th video
        # in current category.
        features = []

        for filename in filenames:
            filepath = os.path.join("..", "dataset", category, filename)
            vid = cv2.VideoCapture(filepath)

            # Store features in current file.
            features_current_file = []

            prev_frame = None

            while vid.isOpened():
                ret, frame = vid.read()
                if not ret:
                    break

                # Only care about gray scale.
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if prev_frame is not None:
                    # Calculate optical flow.
                    flows = cv2.calcOpticalFlowFarneback(prev_frame, frame,
                        **farneback_params)

                    feature = []
                    for r in range(120):
                        if r % 10 != 0:
                            continue
                        for c in range(160):
                            if c % 10 != 0:
                                continue
                            feature.append(flows[r,c,0])
                            feature.append(flows[r,c,1])
                    feature = np.array(feature)

                    features_current_file.append(feature)

                prev_frame = frame

            features.append({
                "filename": filename,
                "category": category,
                "features": features_current_file 
            })

            n_processed_files += 1
            if n_processed_files % 30 == 0:
                print("Done %d files" % n_processed_files)

        # Dump data to file.
        pickle.dump(features, open("data/optflow_%s.p" % category, "wb"))

