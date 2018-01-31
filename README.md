# Action Recognition on the KTH dataset

In this work, we would like to explore different machine learning techniques in recognizing human actions from videos in the KTH dataset. As deep learning is one of the hottest trends in machine learning, our main concentration is to experiment with deep learning models such as the CNN and the LSTM. Moreover, we also examine traditional techniques in computer vision such as Bag-of-Words model, SIFT, and Optical Flow. These traditional techniques are used as the baselines to compare with the deep learning approach.

## KTH dataset
Official web page of KTH dataset: [link](http://www.nada.kth.se/cvap/actions). 
The KTH dataset consists of videos of humans performing 6 types of action: boxing, handclapping, handwaving, jogging, running, and walking. There are 25 subjects performing these actions in 4 scenarios: outdoor, outdoor with scale variation, outdoor with different clothes, and indoor. The total number of videos is therefore 25x4x6 = 600. The videos' frame rate are 25fps and their resolution is 160x120. More information about the dataset can be looked up at the website.

## Methods
We have tried the following approaches:
* SIFT Features + Bag-of-Words + SVM: 47.22%.
* Optical Flow + Bag-of-Words + SVM: 78.24%.
* CNN on single frame: 63.43%.
* CNN on block of frames: 70.37%.

## Prerequisites
* Python OpenCV.
* PyTorch.
* Download the dataset and put the video folders into *dataset/* directory, or you can `cd dataset` and run `./download.sh`.

## Acknowledgement
This was initialized as the final project for our machine learning class in Fall 2017, at the University of Maryland, College Park. The members in our group are: Aya Ismail, Karan Kaur, Khoi Pham (me), Shambhavi Kumar, and Turan Kaan Elgin.