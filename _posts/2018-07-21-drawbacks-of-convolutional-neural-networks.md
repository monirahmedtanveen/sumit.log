---
layout: post
comments: false
title: "Drawbacks of Convolutional Neural Networks"
date: 2018-07-21 10:18:00
tags: computer-vision cnn review
---

>  Although Convolutional Neural Networks has got tremendous success in Computer Vision field, it has unavoidable limitations like it unability to encode Orientational and relative spatial relationships, view angle.  


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

Convolutional Neural Networks(CNN) define an exceptionally powerful class of models. CNN-based models achieving state-of-the-art results in classification, localisation, semantic segmentation and action recognition tasks, amongst others. Nonetheless, they have their limits and they have fundamental drawbacks and sometimes it's quite easy to fool a network. In this post, I rearranged <a href="https://medium.com/ai%C2%B3-theory-practice-business/understanding-hintons-capsule-networks-part-i-intuition-b4b559d1159b">this post</a> from medium to highlight some issues of CNN and add some additional insights.

##  CNN do not encode the position and orientation of  object

Let us consider a very simple and non-technical example. Imagine a face. What are the components? We have the face oval, two eyes, a nose and a mouth. For a CNN, a mere presence of these objects can be a very strong indicator to consider that there is a face in the image. Orientational and relative spatial relationships between these components are not very important to a CNN.
<img src="assets/images/posts/2018-07-21-drawbacks-of-convolutional-neural-networks/face.png" alt="face">
<center> To a CNN, both pictures are similar, since they both contain similar elements</center>

How do CNNs work? The main component of a CNN is a convolutional layer. Its job is to detect important features in the image pixels. Layers that are deeper (closer to the input) will learn to detect simple features such as edges and color gradients, whereas higher layers will combine simple features into more complex features. Finally, dense layers at the top of the network will combine very high level features and produce classification predictions.

An important thing to understand is that higher-level features combine lower-level features as a weighted sum: activations of a preceding layer are multiplied by the following layer neuron’s weights and added, before being passed to activation nonlinearity. Nowhere in this setup there is pose (translational and rotational) relationship between simpler features that make up a higher level feature. CNN approach to solve this issue is to use max pooling or successive convolutional layers that reduce spacial size of the data flowing through the network and therefore increase the “field of view” of higher layer’s neurons, thus allowing them to detect higher order features in a larger region of the input image.

In the example above, a mere presence of 2 eyes, a mouth and a nose in a picture does not mean there is a face, we also need to know how these objects are oriented relative to each other.

In a CNN, all low-level details are sent to all the higher level neurons. These neurons then perform further convolutions to check whether certain features are present. This is done by striding the receptive field and then replicating the knowledge across all the different neurons

CNN do not encode the position and orientation of the object into their predictions. They completely lose all their internal data about the pose and the orientation of the object and they route all the information to the same neurons that may not be able to deal with this kind of information. A CNN makes predictions by looking at an image and then checking to see if certain components are present in that image or not. If they are, then it classifies that image accordingly.


## Lack of ability to be spatially invariant to the input data

In order to correctly do classification and object recognition, it is important to preserve hierarchical pose relationships between object parts. Consider the image below. You can easily recognize that this is the Statue of Liberty, even though all the images show it from different angles. This is because internal representation of the Statue of Liberty in your brain does not depend on the view angle. You have probably never seen these exact pictures of it, but you still immediately knew what it was.
<img src="assets/images/posts/2018-07-21-drawbacks-of-convolutional-neural-networks/Statue-of-Liberty.jpeg" alt="Statue of Liberty">
<center>Your brain can easily recognize this is the same object, even though all photos are taken from different angles. CNNs do not have this capability</center>

For a CNN, this task is really hard because it does not have this built-in understanding of 3D space. In order to learn to tell object apart, the human brain needs to see only a couple of dozens of examples, hundreds at most. CNNs, on the other hand, need tens of thousands of examples to achieve very good performance, which seems like a brute force approach that is clearly inferior to what we do with our brains.

Artificial neurons output a single scalar. In addition, CNNs use convolutional layers that, for each kernel, replicate that same kernel’s weights across the entire input volume and then output a 2D matrix, where each number is the output of that kernel’s convolution with a portion of the input volume. So we can look at that 2D matrix as output of replicated feature detector. Then all kernel’s 2D matrices are stacked on top of each other to produce output of a convolutional layer.

Then, we try to achieve viewpoint invariance in the activities of neurons. We do this by the means of max pooling (e.g. 2 × 2 pixels) that consecutively looks at regions in the above described 2D matrix and selects the largest number in each region. As result, we get what we wanted — invariance of activities. Invariance means that by changing the input a little, the output still stays the same. And activity is just the output signal of a neuron. In other words, when in the input image we shift the object that we want to detect by a little bit, networks activities (outputs of neurons) will not change because of max pooling and the network will still detect the object.

The above described mechanism is not very good, because max pooling loses valuable information and also does not encode relative spatial relationships between features. Because of this, CNN are not actually invariant to large transformations of the input data.


## How to deal with CNN

There are several research to address the issues of CNN. I listed bellow the most promising work.<br>
1.<a href="https://arxiv.org/abs/1710.09829"> Dynamic Routing Between Capsules </a> <br>
2.<a href="https://arxiv.org/abs/1506.02025"> Spatial Transformer Networks </a>


Thanks for reading. I hope you find this post useful.
