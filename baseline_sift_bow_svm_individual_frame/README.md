## SIFT + Bag-of-Words + SVM (frame-level)
In this approach, we consider each frame as an individual instance and we classify each frame instead of a whole video. The majority of all frames' classification results is selected as the answer for a video.

There are 4 main steps in this method:
* SIFT features extraction.
* K-means clustering to build codebook.
* Build Bag-of-Words vector for each frame.
* Train with SVM.

---
### How to run
1. Extract SIFT features from videos.
```bash
$ python extract_sift.py
```
2. Based on the datset splitting instruction on KTH webpage, split the computed SIFT features into train, dev, and test set. This also generates a file `train_keypoints.p` of all SIFT features in the train set whose format is specifically used for clustering.
```bash
$ python make_dataset.py
```
3. Run K-means on `train_keypoints.p` with 1000 as the number of clusters and produce the codebook.
```bash
$ python clustering.py --dataset=data/train_keypoints.p --clusters=1000
```
4. Make BoW vector for every video frame in the training set, using the computed clusters with TF-IDF weighting scheme.
```bash
$ python make_bow_vector.py --codebook=data/cb_1000clusters.p --tfidf=1 --dataset=data/train.p --output=data/train_bow_c1000.p
```
5. Train linear SVM on BoW vectors of training set.
```bash
$ python train_svm.py --dataset_bow=data/train_bow_c1000.p --C=1 --output=data/svm_C1_c1000.p
```
6. Use computed SVM classifier to classify videos in test set and get accuracy result.
```bash
$ python evaluate.py --svm_file=data/svm_C1_c1000.p --bow_file=data/test_bow_c1000.p
```

---
### SIFT Feature Extraction
Because there can be a lot of frames in a video that do not contain human, we use the built-in HOG human detector of OpenCV to look for human in a frame. Having found a human, we draw a bounding box around the human and only compute SIFT features inside this box.

The following table shows the number of frames per category containing human in the training set. Note that these are only the frames that were detected by the OpenCV's built-in HOG human detector. There is limitation in this detector because it was not trained on the KTH dataset. There are a lot of frames in categories with fast moving action such as running and jogging that the human was failed to be detected.

| Category       | # frames human detected | # SIFT keypoints |
| -------------- |:-----------------------:|:----------------:|
| boxing         | 5892                    | 189309           |
| handclapping   | 6978                    | 253708           |
| handwaving     | 7984                    | 286754           |
| jogging        | 1327                    | 46729            |
| running        | 623                     | 20901            |
| walking        | 2375                    | 77044            |

The total number of keypoints is enormous, thus we randomly sample a portion of them for training. To keep the training data balanced between classes, we randomly sample 10000 keypoints in each category and use them to build our codebook.

---
### Training & Model Selection
We use K-means clustering with different number of clusters. These clusters are used for vector quantization and building Bag-of-Words (BoW) vector for each frame. We then train a linear SVM classifier on this set of BoW vectors. Experimental results show that linear SVM classifier produce better results than SVM with other types of kernels.

The validation set is used for evaluating our model with different configurations. The best hyperparameters are:
* Number of clusters: 1000.
* Use TF-IDF weighting scheme for BoW.
* Type of SVM kernel: linear.
* Penalty C of linear SVM classifier: 1.

---
### Results
Accuracy on the test set is only 47.22%.
Three potential problems could be:
* SIFT feature does not seem to capture the distinction between activities. 
* Classification of a video is based on the majority voting of its frames. This ignores the temporal relation between frames.
* Built-in HOG human detector of OpenCV is unable to detect human in a lot of frames in this dataset.