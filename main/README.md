## Deep Neural Network
Different videos vary in length with different number of frames, therefore, it is not easy to use a fixed-sized deep neural network to handle a whole video. As video is a sequence of frames, we can create a model that extends in the time domain, processes a fixed-length contiguous sequence of frames, so that it can learn spatio-temporal features. We trained 3 models, each of which is different in the number of video frames that it handles as input + some additional features.
* CNN on single frame.
* CNN on block of contiguous frames.
* CNN on block of contiguous frames + optical flow features.

### Prepare Data
Before starting to train, we ought to prepare the data. Run the following python file to create the dataset for training our models.
```bash
$ python data_utils.py
```

### CNN Single Frame
This model is only trained on individual frame. The goal of training this model is to quantify the importance of visual features from each individual frame in determining the video's label.

To start training this model with batch size 64, 20 epochs, and no CUDA, run
```bash
$ python train_cnn_single_frame.py --batch_size=64 --start_epoch=1 --num_epochs=20 --cuda=0
```
The trained models after each epoch are saved at `data/cnn_single_frame` with the names in the following format: `model_epoch%num_epoch`.

To resume training from a model that has been trained for 20 epochs for another 20 epochs, run
```bash
$ python train_cnn_single_frame.py --batch_size=64 --start_epoch=21 --num_epochs=20 --cuda=0
```

To classify a video, we run this model on each individual frame to get the frame's vector of class probabilities. We take the average of all probabilities vectors to get the final class probabilities for the video. To evaluate a trained model on the test set, run
```bash
$ python eval_cnn_single_frame.py --model_dir=data/cnn_single_frame/model_epoch22.chkpt
```

The accuracy of this model on the test set is 63.43%. This means that each activity's visual features are quite different from the other. This beats the baseline SIFT + BoW method by a large margin.

### CNN Block of Frames
For each video, we devide it into blocks of 15 contiguous frames. The model is then trained on these blocks instead of individual frame. In the convolutional layers, we use 3D convolutional filters to train the model to learn to detect temporal features. Running the code to train this model is similar to that of the CNN single frame model. Just replace the python file with `train_cnn_block_frame.py` and that's it.

To classify a video, we also divide it into blocks of 15 contiguous frames. We then run the model on each block to get the block's vector of class probabilities. These vectors are also averaged so that we can get the final class probabilities of the video. To evaluate the model on the test set, run `eval_cnn_block_frame.py` just like in the last section.

This model's accuracy on the test set is 70.37%. This means that the model is able to detect motion features in consecutive frames. However, its accuracy is nowhere near the baseline optical flow + BoW method.

### CNN Block of Frames + Optical Flow
This model is the same as the last one, except that it is additionally trained on optical flow features. Run `train_cnn_block_frame_flow.py` to start training and run `eval_cnn_block_frame_flow.py` to evaluate the trained model on the test set.

The accuracy of this model on the test set is 90.27%. This totally outperforms the previous methods.
