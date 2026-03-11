Thank you reviewing our paper.

#Environment setup:
##Required libraries and versions:
Python 3.8.10       : pip install python==3.8.10
Numpy 1.24.3        : pip install numpy==1.24.3
Tensorflow 2.13.0   : pip install tensorflow==1.13.0
Keras 1.13.1        : pip install keras==1.13.1
PIL                 : pip install pillow 

#Sample test image and attack setup:
    test image: "baseball3.JPEG" as clean image. Its size is: (h x w)=(2336 x 3504)
    target category: "ladle", its index number is 421 in the ImageNet dataset
    target label value: 0.9
    attacked model: pre-trained ResNet-50 model (trained on ImageNet dataset)
    epsilon: 4  # maximum allowed pixel magnitude change [-epsilon:epsilon]

#Running:
You can simply run NbuGAN.py in any IDE.
During the process, it will report the changes in image label and its value for each epoch. 
At the end, the generated adversarial image will be saved in .png and .npy formats. 
The "Report.txt" will provide information about the results, 
including the label and label value of the adversarial image, the number of epochs, and the time taken in seconds.



