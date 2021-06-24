# EVA6/ Session 7

Our Team members are

1. Pratima Verma

2. L.Mahesh

Session 7 Assignment
Problem Statement

Change the code such that it uses GPU
Change the architecture to C1C2C3C40 (No MaxPooling, but 3 3x3 layers with stride of 2 instead) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)
Total RF must be more than 44
One of the layers must use Depthwise Separable Convolution
One of the layers must use Dilated Convolution
Use GAP (compulsory):- add FC after GAP to target #of classes (optional)
Use albumentation library and apply:
Horizontal flip
ShiftScaleRotate
CoarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
Achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.

Assignment S7 brief

The Code flow will be as below

Data-> Dataset-> DataLoader-> Model-> Loss-> Optimizer

Package is torchvision
Data is CIFAR10.
Data loaders to be used is torch.utils.data.DataLoader
CIFAR10 classes: ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’.
The images in CIFAR-10 are of size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size.

Approach steps
GPU conversion and verification
Load and normalizing the CIFAR10 training and test datasets using torchvision
Define a Convolution Neural Network
Define a loss function
Train and Test the network
Model Plot for the Accuracy

Model Summary

Requirement already satisfied: torchsummary in /usr/local/lib/python3.7/dist-packages (1.5.1)
cuda

Total params: 96,352

Trainable params: 96,352

Non-trainable params: 0

Accuracy of the network on the 10000 test images: 80.620 %

EPOCHS = 30

Learning Rate lr=0.018
