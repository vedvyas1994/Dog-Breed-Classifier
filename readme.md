# Deep Learning

## Capstone Project : Dog Breed Classifier using CNN (Convolutional Neural Networks)

### By Vedavyas Kamath


### Goal/Aim
The goal of this project is to explore the deep learning approach to classify images, in this case, dog breeds using a type of Neural Network called the [Convolutional Neural Network (CNN)](http://cs231n.github.io/convolutional-networks/). The classifier should be able to detect the prescence of a human face or a dog face in the image and then return the dog breed with highest resemblence.


### Data
Worked with 2 separate datasets :
1. Dog Image Dataset available for download [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). This needs to be placed at the `/dog_images` directory in project's home directory.
2. Human Image Dataset available for download [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip). And this needs to be placed at the `/lfw` directory in project's home directory.


### Project Design & Overview

This project has been completed under a single Jupyter notebook `dog_app.ipynb`

I have followed the below design/workflow to be able to solve this problem of dog breed classification in a step-by-step manner through the course of this project.

#### 1. Data setup: 
*a. Download* : This step is not applicable as the data is already present on Udacity workspace. <br>
*b. Loading* : Will load this data by creating a custom loader to load training, validation and test sets. <br>
*c. Pre-processing*: To convert the image file into a tensor that can be used by CNN for classification. <br>
        
#### 2. Human face detection:        
*a. OpenCV for Human Face Detection* : Will use OpenCV's implementation of Haar feature-based cascade classifiers to detect human faces in images. 

#### 3. Dog's detection: 
*a. VGG16* : Use Pre-trained network VGG16 to detect presence of any dog in the image. <br>

#### 4. Dog's breed classifier: 
     
***1. Create a CNN to Classify Dog Breeds (from Scratch):***
- Here I plan to define a CNN from scratch by deciding the different types and number of layers that will be used in my network to classify the breed. 
- I then plan to make it feed forward by defining the overall path in which data will be propagted through the network to finally arrive to any one of the 133 possible dog breed classes. 
		
***2. Use transfer learning to use the already trained models (resnet50) as a base for our model:*** 
- Pre-trained networks are efficient and well suitable to solve challenging problems because once trained, these models work as feature detectors even for images they werent trained on. 
- Here I will use transfer learning to train a pre-trained network (resnet50) which is trained on ImageNet and readily available for use in torchvision.models so that it is able to classify our dog images. I have used the "resnet50" model    

#### 5. Write an algorithm:
***The final algorithm accepts input as image and does the following :***
   1. applies pre-processing to the input image,
   2. checks for a dog's face and returns its breed if found
   3. checks for human face, and if found returns the dog breed which human's face resembles to.
   4. Handles the error in an interactive manner if neither dog or human face is found in the image.

See the project report [Project_report_Dog_Breed_Classifier.pdf](https://github.com/vedvyas1994/Dog-Breed-Classifier/blob/master/Project_report_Dog_Breed_Classifier.pdf) for more details about the project.

### Software Requirements
This project requires **Python 3.x** and the following Python libraries installed:

- [Jupyter notebook](https://jupyter.org/) / [AWS Sagemaker Notebook](https://aws.amazon.com/sagemaker/), Jupyter Notebook is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations and narrative text. It was used from data exploration to final algorithm tests
- [PyTorch](https://pytorch.org/), open source machine learning library based on the Torch library. It was used to developer the CNN models to predict the dog's breed
- [OpenCV](https://opencv.org/), open source computer vision and machine learning software library. It was used to identify the human face
- [Matplotlib](https://matplotlib.org/), plotting library which produces publication quality figures. It was used to show images and plot graphs
- [NumPy](https://numpy.org/), library which adds support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

NOTE: <br>
This dataset is quite complex and so it is reccomended to utilize a GPU for training.

### References:
TO learn about OpenCV's Cascade Classifiers: <br>
https://docs.opencv.org/trunk/db/d28/tutorial_cascade_classifier.html <br>

For converting an image to gray-scale: <br>
https://techtutorialsx.com/2018/06/02/python-opencv-converting-an-image-to-gray-scale/ <br>

TO learn about the various pre-trained models: <br>
https://pytorch.org/docs/master/torchvision/models.html <br>

TO learn about the architecture of CNN and its different components & uses: <br>
https://easyai.tech/en/ai-definition/cnn/
https://analyticsindiamag.com/convolutional-neural-network-image-classification-overview/

Architecture and Working of VGG16 Model: <br>
https://neurohive.io/en/popular-networks/vgg16/

Working of OpenCV's HAAR Classifiers: <br>
https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
https://becominghuman.ai/face-detection-using-opencv-with-haar-cascade-classifiers-941dbb25177

Resnet-50 model: <br>
https://pytorch.org/hub/pytorch_vision_resnet/
https://missinglink.ai/guides/pytorch/pytorch-resnet-building-training-scaling-residual-networks-pytorch/
