# CNN-Architectures-from-LeNet-to-AlexNet-1998-2012-
Since the 1950s, a small subset of Artificial Intelligence (AI),often called Machine Learning (ML), has revolutionized
several fields in the last few decades. Neural Networks(NN) are a subfield of ML, and it was this subfield that spawned
Deep Learning (DL)

It all started with LeNet in 1998 and eventually, after nearly 15 years, lead to ground breaking models winning the ImageNet Large Scale Visual Recognition Challenge which includes AlexNet in 2012 to GoogleNet in 2014 to ResNet in 2015 to ensemble of previous models in 2016. In the last 2 years, no significant progress has been made and the new models are an ensemble of previous ground breaking models.
# LeNet 5
LeNet5 is a small network, it contains the basic modules of deep learning: convolutional layer, pooling layer, and full link layer. It is the basis of other deep learning models. Here we analyze LeNet5 in depth. At the same time, through example analysis, deepen the understanding of the convolutional layer and pooling layer.
Average pooling
- Sigmoid or tanh nonlinearity
- Fully connected layers at the end
- Trained on MNIST digit dataset with 60K training examples 
- LeNet-5 is a very efficient convolutional neural network for handwritten character recognition.
- Convolutional neural networks can make good use of the structural information of images.
- The convolutional layer has fewer parameters, which is also determined by the main characteristics of the convolutional layer, that is, local connection and shared weights.

## Authors: Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner
Published in: Proceedings of the IEEE (1998)
### Source: R. Fergus, Y. LeCun
![](https://engmrk.com/wp-content/uploads/2018/09/LeNEt_Summary_Table.jpg)
![](https://miro.medium.com/max/1072/0*H9_eGAtkQXJXtkoK)
![](https://ujwlkarn.files.wordpress.com/2016/08/car.png?w=1212)
#### Over the next several years, we worked to improve our computational power so that we can do a large number of calculations in reasonable time and overcome the limitations faced by LeNet.
#### The quest has been successful and this led to a research into deep learning and gave rise to ground breaking models.
## IMAGET COMPETITION (Fast forward to the arrival of big visual data…)
- ~14 million labeled images, 20k classes
- Images gathered from Internet
- Human labels via Amazon MTurk
- ImageNet Large-Scale Visual Recognition Challenge (ILSVRC):1.2 million training images, 1000 classes 
![](https://i0.wp.com/syncedreview.com/wp-content/uploads/2020/06/Imagenet.jpg?fit=1400%2C600&ssl=1)
### http://www.image-net.org/challenges/LSVRC/
# AlexNet ILSVRC 2012 winner 
AlexNet was designed by Hinton, winner of the 2012 ImageNet competition, and his student Alex Krizhevsky. It was also after that year that more and deeper neural networks were proposed, such as the excellent vgg, GoogleLeNet. Its official data model has an accuracy rate of 57.1% and top 1-5 reaches 80.2%. This is already quite outstanding for traditional machine learning classification algorithms.
Similar framework to LeNet but:
- Max pooling, ReLU nonlinearity
- More data and bigger model (7 hidden layers, 650K units, 60M params)
- GPU implementation (50x speedup over CPU)
- Trained on two GPUs for a week
- Dropout regularization
## Authors:A. Krizhevsky, I. Sutskever, and G. Hinton, ImageNet Classification with Deep Convolutional Neural Networks, NIPS 2012 
### Source:
![](https://miro.medium.com/max/1372/0*xPOQ3btZ9rQO23LK.png)
# Clarifai: ILSVRC 2013 winner
###  Refinement of AlexNet
##### M. Zeiler and R. Fergus, Visualizing and Understanding Convolutional Networks,ECCV 2014 (Best Paper Award winner) 
![](https://software.intel.com/content/dam/develop/external/us/en/images/dev-journey-article16-fig3-zfnet-architecture-747495.png)
## Why does AlexNet achieve better results?
1. **Overlapping Max Pooling**
Overlapping Max Pool layers are similar to Max Pool layers except the adjacent windows over which the max is calculated overlaps each other. The authors of AlexNet used pooling windows, sized 3×3 with a stride of 2 between the adjacent windows. Due to this overlapping nature of Max Pool, the top-1 error rate was reduced by 0.4% and top-5 error rate was reduced by 0.3% respectively. If you compare this to using a non-overlapping pooling windows of size 2×2 with a stride of 2, that would give the same output dimensions.
![](https://blog.kakaocdn.net/dn/b5hfOx/btqBCUY3kpE/CKcK19bmDgtkSkWS5GPkBk/img.png)
1. **Relu activation function is used.**
- Relu function: f (x) = max (0, x)
ReLU-based deep convolutional networks are trained several times faster than tanh and sigmoid- based networks. The following figure shows the number of iterations for a four-layer convolutional network based on CIFAR-10 that reached 25% training error in tanh and ReLU
![](https://image.slidesharecdn.com/ucl-irdm-deeplearning-160429080538/95/deep-learning-39-638.jpg?cb=1461917930)
2. **Standardization ( Local Response Normalization )**
- After using ReLU f (x) = max (0, x), you will find that the value after the activation function has no range like the tanh and sigmoid functions, so a normalization will usually be done after ReLU, and the LRU is a steady proposal (Not sure here, it should be proposed?) One method in neuroscience is called "Lateral inhibition", which talks about the effect of active neurons on its surrounding neurons.
![](https://qph.fs.quoracdn.net/main-qimg-5121f4aac5305e50dbe19bf1aeab0a90)
3. **Dropout**
- Dropout is also a concept often said, which can effectively prevent overfitting of neural networks. Compared to the general linear model, a regular method is used to prevent the model from overfitting.
During dropout, a neuron is dropped from the Neural Network with a probability of 0.5. When a neuron is dropped, it does not contribute to forward propagation or backward propagation. Every input goes through a different Neural Network architecture, as shown in the image below. As a result, the learned weight parameters are more robust and do not get overfitted easily.
4. **Enhanced Data ( Data Augmentation )**
- When you show a Neural Net different variation of the same image, it helps prevent overfitting. It also forces the Neural Net to memorize the key features and helps in generating additional data. 

Data Augmentation by Mirroring

Let’s say we have an image of a cat in our training set. The mirror image is also a valid image of a cat. This mean that we can double the size of the training datasets by simply flipping the image above the vertical axis.
![](https://cdn-images-1.medium.com/max/1013/1*rvwzKkvhlDN3Wo_4Oay_4Q.png)
# Results 
In the 2010 version of ImageNet challenge AlexNet vastly outpaced the second-best model with 37.5% top -1 error vs 47.5% top-1 error , and 17.0% top-5 error to 37.55 top-5 error. AlexNet was able to recognize off-center objects and most of its top 5 classes for each image were reasonable. AlexNet won the 2012 competition with a top-5 error rate of 15.3% compared to second place top-5 error rate of 26.2%.
## Chan info
Shortly after winning the ImageNet competition Alex Krizhevsky and Ilya Sutskever sold their startup DNN research Inc to Google. Alex worked in Google till 2017 when he left Google (citing loss of interest) to work at Dessa where he will advise and research new deep learning techniques. Ilya Sutskever left Google in 2015 to become director of the OpenAI Institute and is currently working at the same place.
### Source: Google,Internet,ImageNet
