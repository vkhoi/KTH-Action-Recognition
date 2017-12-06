import cv2
import os
import pickle

CATEGORIES = ["boxing", "handclapping", "handwaving", "jogging", "running", 
    "walking"]

if __name__ == "__main__":

    # Create directory to store extracted SIFT features.
    os.makedirs("data", exist_ok=True)

    # Setup HOG descriptor to detect human.
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Setup SIFT feature detector.
    sift = cv2.xfeatures2d.SIFT_create()

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

            while vid.isOpened():
                ret, frame = vid.read()
                if not ret:
                    break

                # Only care about gray scale.
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect human.
                (rects, weights) = hog.detectMultiScale(frame, winStride=(1,1),
                    padding=(8,8), scale=1.1)

                # Check if human is detected.
                if len(rects) > 0:
                    # Assume that there's only one human in a frame.
                    # Construct bounding box's coordinates.
                    x0, y0, w, h = rects[0]
                    x1 = x0 + w
                    y1 = y0 + h

                    human_image = frame[y0:y1, x0:x1]

                    # Detect SIFT keypoints and compute SIFT descriptors.
                    kps = sift.detect(human_image)
                    desc = sift.compute(human_image, kps)

                    # Append features.
                    if desc[1] is not None:
                        features_current_file.append(desc[1])
                
            features.append({
                "filename": filename,
                "category": category,
                "features": features_current_file 
            })

            n_processed_files += 1
            if n_processed_files % 30 == 0:
                print("Done %d files" % n_processed_files)

        # Dump data to file.
        pickle.dump(features, open("data/sift_%s.p" % category, "wb"))

