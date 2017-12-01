## Optical Flow + Bag-of-Words + SVM
There are 4 main steps in this method:
* Optical flow for consecutive frames in videos.
* K-means clustering to build codebook.
* Build Bag-of-Words vector for each video.
* Train with SVM.

---
### How to run
1. Extract optical flow in the x and y direction for each frame in all videos.
```bash
$ python extract_optical_flow.py
```
2. Based on the datset splitting instruction on KTH webpage, split the computed optical flow features into train, dev, and test set. This also generates a file `train_keypoints.p` of all optical flow features in the train set whose format is specifically used for clustering.
```bash
$ python make_dataset.py
```
3. Run K-means on `train_keypoints.p` with 200 as the number of clusters and produce the codebook.
```bash
$ python clustering.py --dataset=data/train_keypoints.p --clusters=200
```
4. Make BoW vector for every video in the training set, using the computed clusters with TF-IDF weighting scheme.
```bash
$ python make_bow_vector.py --codebook=data/cb_200clusters.p --tfidf=1 --dataset=data/train.p --output=data/train_bow_c200.p
```
5. Train linear SVM on BoW vectors of training set.
```bash
$ python train_svm.py --dataset_bow=data/train_bow_c200.p --C=1 --output=data/svm_C1_c200.p
```
6. Use computed SVM classifier to classify videos in test set and get accuracy result.
```bash
$ python evaluate.py --svm_file=data/svm_C1_c200.p --bow_file=data/test_bow_c200.p
```

---
### Optical Flow Computation
Given a current frame and its previous frame, we can compute its optical flow feature using the built-in dense optical flow Gunnar Farneback's algorithm of OpenCV. Thus, given a video with N frames, we can compute a set of N-1 optical flow feature descriptors.

As the videos' resolution are 160x120 and we want to save memory, we only sample the optical flow values on the rows and columns whose indices are multiples of 10 (i.e. row and columnn 0, 10, 20, ...). The optical flow descriptor for a frame will have size 16x12x2 = 384 (2 comes from the horizontal and vertical direction).

---
### Training & Model Selection
We use K-means clustering with different number of clusters. These clusters are used for vector quantization to assign each optical flow descriptor to its nearest codeword.

Next, we build Bag-of-Words (BoW) vector for each video. A BoW vector is like a histogram that counts the frequency of optical flow descriptors which appear in a video. We also perform TF-IDF weighting scheme since it leads to better accuracy. We then train a linear SVM classifier on our training set. As the number of videos in our training set is only 192, the training process takes less than a second, which is a lot faster than the method of considering each individual frame as an instance.

The validation set is used for evaluating our model with different configurations. The best hyperparameters are:
* Number of clusters: 200.
* Use TF-IDF weighting scheme for BoW.
* Type of SVM kernel: linear.
* Penalty C of linear SVM classifier: 1.

---
### Results
Accuracy on the test set is 78.24%. This result outperforms the other method of using SIFT feature and considering each individual frame as an instance, which only achieves 47.22%.
