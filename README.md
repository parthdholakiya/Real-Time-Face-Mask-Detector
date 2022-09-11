# Real-Time-Face-Mask-Detector
Build a real-time system to detect whether the person on the webcam is wearing a mask or not. We will train the face mask detector model using Keras and OpenCV.

We will build a real-time system to detect whether the person on the webcam is wearing a mask or not. We will train the face mask detector model using Keras and OpenCV.

Download the Dataset
The dataset we are working on consists of 1376 images with 690 images containing images of people wearing masks and 686 images with people without masks.

Download the dataset: https://data-flair.training/blogs/download-face-mask-data/

![image](https://user-images.githubusercontent.com/94167271/189524758-45d99e2b-b815-4e18-9300-01123bbb13f4.png)


##### Build the convolution neural network:

This convolution network consists of two pairs of Conv and MaxPool layers to extract features from the dataset. Which is then followed by a Flatten and Dropout layer to convert the data in 1D and ensure overfitting.

And then two Dense layers for classification.

model = Sequential([
    Conv2D(100, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2,2),
    
    Conv2D(100, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dropout(0.5),
    Dense(50, activation='relu'),
    Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

##### Image Data Generation/Augmentation:

Data augmentation techniques improve the accuracy of computer vision models. Using additional images during the training phase adds variety and more features to your existing data, which your model can use to generalize more and reduce overfitting.

Data augmentation in data analysis are techniques used to increase the amount of data by adding slightly modified copies of already existing data or newly created synthetic data from existing data. It acts as a regularizer and helps reduce overfitting when training a machine learning model.

![image](https://user-images.githubusercontent.com/94167271/189523214-a566bc6a-e652-4cef-ae84-f6e246e19fc3.png)


 Initialize a callback checkpoint to keep saving best model after each epoch while training:
 
 ### model reach upto 98% val_acc
 
 ![Screenshot (235)](https://user-images.githubusercontent.com/94167271/189524311-0cc068fd-c407-4efe-a904-32a0c5ca4961.png)

 
 
 ### the results of face mask detector model using OpenCV.

![image](https://user-images.githubusercontent.com/94167271/189524475-d5b8dacf-a282-4529-a972-6ed03d4b1702.png)

 
 
 
 
