# **Behavioral Cloning** 

## Writeup

This writeup summarizes my solution to the behavioral cloning task in the udacity self-driving nanodegree. 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

<a href="http://www.youtube.com/watch?feature=player_embedded&v=vMSE24ygjIc
" target="_blank"><img src="http://img.youtube.com/vi/vMSE24ygjIc/0.jpg" 
alt="neural net driving on both tracks" width="240" height="180" border="10" /></a>
https://youtu.be/vMSE24ygjIc

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image3]: ./images/steering_angles.png "Distribution of steering angles"
[image4]: ./images/random-frames.png "Random frames"
[image6]: ./images/loss.png "loss diagram"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I focused my work on both the nvidia model (https://devblogs.nvidia.com/deep-learning-self-driving-cars/) and the commaai-architecture https://github.com/commaai/research. 
Both models contain convolutions followed by fully connected layers. The nvidia-model features 5 convolutions, followed by 3 fully connected layers. The commaai-model uses 3 convolutions, followed by 2 fully connected layers, thus uses less parameters.
Both the nvidia and the commaai-model are originally designed to calculate a steering angle using a videostream. 

#### 2. Attempts to reduce overfitting in the model

The weights on both nets are initialized according to the orthogonal algorithm. This algorithm is designed to prevent linear dependancy in the matrices. 
The model further contains dropout layers in order to reduce overfitting within the last convolutional layer of the nvidia model and all fully connected layers.  
The training data is randomly split into 80 % training and 20 % validation data. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. The more augmentation is used, the more epochs are useful. The final model was created using 7 epochs. 

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used strict center-lane driving. Known difficult turns were driven slowly, in order to have them well-represented in the dataset. 
The udacity dataset was used, and enriched by one lap of driving on the lakeside-track. Three normal rounds of the jungle track were recorded. Three further laps started at the second half of the track, to record the difficult section twice per lap. 
All records were done using a mouse. 
For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to focus on the dataset and it's augmentation. 

My first step was to improve the LeNet architecture. After some time this was discarded to favor the nvidia architecture, which was proven to work on an appropriate dataset. I also included the similar, yet smaller commaai network, because of the sparse dataset. Driving back and forth on the track, using different graphics levels still produces strongly correlated data. Due to it's compact design, the commaai model is less prone to overfit. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. Usually, overfitting wasn't an issue, since I initially used a lot of driving data with a dataset of 60000 frames. 
Using a dataset with center-lane, left-lane and right-lane driving, the least mean square error achievable increases. The driving itself shows improved stability. Yet, the model learns to drive on the side of the road, and slight mistakes result in immediate crashes. 

Using a small dataset of 10.000 raw frames (plus augmentation) showed, that the model repeats errors of the human. Also, not the entire track was driven correctly. Thus, a clean dataset of 27000 frames was created. 
The model then usually sticked to the center, but it got sidetracked by partially shadowed road sections. Using the same model on low graphics settings, the issue was resolved. 
In order to train a model which doesn't bother about shadows, both the complete image was randomly darkened and a random-shadow function got implemented. This enabled the model to mostly ignore shadows. 

The lakeside-track proved to have two main difficulties. There is a section in which the road splits into dirt. There, the model initially lost direction. A corner facing the lake also proved it's difficulties. Both issues were resolved by creating more training data. 

The jungle track is quite challenging. Tight sections with shadows are quite difficult to learn. The sharp 180° turn with another road section in the background is the key corner to master. Almost all models failed at this point. Given the camera's perspective, there seems to be a road section in the background connected to the tarmac ahead. This corner was individually recorded a few times. 
Analyzing the training set, there is a strong bias towards driving straight. Managing tight corners, especially the most difficult one, is possible with randomly discarding frames with straight driving. Dropping many frames from the dataset means the network loses information about navigating straight. 
Furthermore, a strong factor is the steering correction due to the side-cameras. Given a clean dataset, the side cameras are always facing towards the side of the road. Using a higher steering angle correction factor teaches the network to correct stronger in case it's drifting towards the side. If the factor is too high, the steering becomes nervous and the pictures don't match anymore, resulting in a higher loss after the same training. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.
Interestingly, the modified nvidia architecture used on the same dataset and same parameters had issues with overfitting and crashed into a lambpost. This issue needs to be adressed in future experiments. 

#### 2. Final Model Architecture

The final model starts with normlization and cropping, followed by three convolutional layers and two fully connected layers with dropout. It is almost identical to the original found here: https://github.com/commaai/research
The input image has a shape of 160x320x3 RGB. It is normalized and cropped to 90x320x3 RGB. The three consecutive convolutional layers search for patterns in the image. The original model features only one fully connected layer, for testing a second fully connected layer was implemented and never removed. 
The standard initializing function was replaced by orthogonal initialization. In theory, the weights should be initialized orthogonally, thus using the number of nodes better. The initial error and the learning rate improved quite a lot due to this change. 
![alt text][image1]

    
#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. It is important to stick to the center lane and use the mouse for continous inputs. The difficult corners were driven slowly. The simulator saves 10 fps constantly, thus slower driving results in a better representation of the data set. 
Then I repeated this process on track two in order to get more data points.
![random frames][image4]

To augment the data set, I also flipped images and angles in order to balance the sides. It also helps to generalize, since the same objects have the same meaning on both sides. I used the extra cameras to further generalize and to generate data for recovery manouvers. The factor was tweaked to 0.3, in order to recover strongly, but still drive quite stable. 
The general brightness of the image gets randomized, and a random shadow helps the model to ignore shadows on the road. 

The dataset consists of 27000 frames. But, straight driving is overly represented, as indicated in the histogram below. 

![histogram][image3]

Whereas the straight driving pictures contain valuable information about the sorrounding elements, the driving behavior is dominated by them. Thus, 66 % of the pictures containing straight driving were dropped to generate the orange dataset. Using random flipping, right and left steering angles get balanced. 

I finally randomly shuffled the data set and put 20 % of the data into a validation set. The data gets shuffled within every batch generated. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7 as evidenced by the error rate below. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![loss graph][image6]
