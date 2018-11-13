---
layout: post
comments: false
title: "Demystify Capsule Network Using Pytorch"
date: 2018-07-21 10:18:00
tags: computer-vision cnn review
---

>  A major improvement on Convolutional Neural Network. Capsule Network more closely mimic biological neural organization and encode better hierarchical relationships.


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

## Introduction
CapsNet is inspired from Rendering process in Computer Graphics and tries to mimic Human Vision system based on an assumption. 

In rendering, a scene file contains some instantiation parameters representing object's geometry, viewpoint, texture, lighting and shading information. Then these parameters are passed through a rendering program to output a digital image into the world. Where human vision system perceive an object using sequence of [fixation points](https://en.wikipedia.org/wiki/Fixation_%28visual%29). In the paper, authors assumed a single fixation point gives us much more than just a single identified object and its properties and our multi-layer visual system creates a parse tree-like structure on each fixation. The paper ignores the issue of how these single-fixation parse trees are coordinated over multiple fixations. Authors proposed a multi-layer neural architecture, where capsules works as fixation points by reversing the rendering process called inverse rendering. These capsule try to encode the instantiation parameters of presence, texture and spatial imformation given a object from its digital image. 

> A capsule is a group of neurons whose activity vector represents the instantiation parameters of a specific type of entity such as an object or an object part. We use the length of the activity vector to represent the probability that the entity exists and its orientation to represent the instantiation parameters. 

## CapsNet Architecture
![Capsule Network]({{ '/assets/images/posts/2018-11-13-demystify-capsule-network-using-pytorch/model_architecture.png' | relative_url }})

Capsule Network has 2 part, Encoder and Decoder. Both consist of three layer.
- [Encoder](#encoder) <br>
    - Layer 1 - [Convolution layer](#convolution-layer)<br>
    - Layer 2 - [Primary Capsule layer](#primary-capsule-layer)<br>
    - Layer 3 - [Digit Capsule layer](#digit-capsule-layer)<br>
- [Decoder](#decoder)<br>
    - Layer 4 - [Fully Connected #1](#fully-connected-#1)<br>
    - Layer 5 - [Fully Connected #2](#fully-connected-#2)<br>
    - Layer 6 - [Fully Connected #3](#fully-connected-#3)<br>

First import necessary libraries.


```python
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import matplotlib.pyplot as plt
```

## Encoder
Encoder part of the network takes as input a 28x28 MNIST digit image and learns to encode it into 10x16-dimensional vectors as output of Digit Capsule layer. Where each 16 dimensional vector represents a capsule for each digit. This vectors are the instantiation parameters of the digits.

Lats start with some random input image (28×28 pixels, 1 color channel = grayscale) and label, like MNIST.
- Pytorch convention -<br>
     Conv Layer input size $$(N,C_{in},H,W)$$ and output size $$(N,C_{out},H_{out},W_{out})$$ <br>
     here, N = # of sample, C = # of channel, H = height, W = width


```python
batch_size = 5
input_images = np.random.rand(batch_size, 1, 28, 28)
input_images = torch.from_numpy(input_images).float() # convert to pytorch tensor.
labels = np.random.randint(0, 10, batch_size)
print('image size - ', input_images.size())
print('labels - ', labels)
```

    image size -  torch.Size([5, 1, 28, 28])
    labels -  [8 4 2 6 1]
    

### Convolution layer 
Now lets define the first Convolution layer with parameter mentioned in the paper and feed the input images. This layer will detect basic features in the image like straight edges, simple colors and curves. In the paper, the convolutional layer has 256 feature maps with kernel size of 9x9, stride 1 and zero padding, followed by non-linear activation ReLU and output a tensor of size 256x20x20.


```python
conv_layer = nn.Conv2d(in_channels=1, out_channels=256, 
                  kernel_size=9, stride=1, padding=0)
print('Weight matrix size- ', conv_layer.weight.data.size())

conv_layer_out = F.relu(conv_layer(input_images))
print('Output size - ', conv_layer_out.size())
```

    Weight matrix size-  torch.Size([256, 1, 9, 9])
    Output size -  torch.Size([5, 256, 20, 20])
    

- Calculate output of convolution layer <br>
$$ output \ height = \dfrac{height - kernel\_size + 2 * padding}{stride} + 1$$ <br>
$$ output \ width = \dfrac{width - kernel\_size + 2 * padding}{stride} + 1$$

## Primary Capsule layer
In this layer, we replace the scaler-output feature detector of CNN with 8-dim vector output capsule for inverse rendering. Each capsule represents every location or entity in the image and encodes different instantiation parameter such as pose (position, size, orientation), deformation, velocity, albedo, hue, texture, etc. If we make slight changes in the image, capsules values also changes accordingly. This is maintained throughout the network. This is called Equivarience. Traditional CNN fails to encode these feature due to the nature of scalar-output feature detector and pooling layers.    

This layer can be designed using several Convolution layer. In the paper, authors used a stack of 8 Convolution layer each with 32 feature maps, kernel size of 9x9, stride 2 and zero padding. We pass the output of the first convolution through every convolution in this layer and our expected final output is $$[batch\_size, primary\_num\_capsule, primary\_capsule\_dim]$$. Here each capsules dimension should be 8 which is actually equal to the number of convolution layer in this layer. Initially we will get output tensor of shape $$[batch\_size, primary\_num\_conv, C_{out}, H_{out}, W_{out}]$$. So we need to reshape the initial output shape to get our expected shape.

> In total Primary Capsules has [32 × 6 × 6] capsule outputs (each output is an 8D vector) and each capsule in the [6 × 6] grid is sharing their weights with each other.  


```python
primary_capsule_dim = primary_num_conv = 8
num_feature = 32
primary_num_capsule = 6 * 6 * num_feature # 1152 primary capsules
```


```python
conv_stack = nn.ModuleList([nn.Conv2d(in_channels=256, 
                         out_channels=num_feature, 
                         kernel_size=9, stride=2, padding=0)
                         for _ in range(primary_num_conv)])
print('Weight matrix shape - ', conv_stack[0].weight.data.size())
```

    Weight matrix shape -  torch.Size([32, 256, 9, 9])
    


```python
primary_capsule_out = [conv(conv_layer_out) for conv in conv_stack]
print('Output shape of every conv layer - ', primary_capsule_out[0].size())

primary_capsule_out = torch.stack(primary_capsule_out, dim=1)
print('Initial output shape - ', primary_capsule_out.size())

primary_capsule_out = primary_capsule_out.view(primary_capsule_out.size(0), -1, primary_capsule_dim)
print('Final output shape - ', primary_capsule_out.size())
```

    Output shape of every conv layer -  torch.Size([5, 32, 6, 6])
    Initial output shape -  torch.Size([5, 8, 32, 6, 6])
    Final output shape -  torch.Size([5, 1152, 8])
    

The length of the output vector of a capsule presents the probability of the entity present in current input that particular capsule is looking for and the orientation of this output vector estimates pose parameters. To achieve this functionality we apply a non-linear 'squashing' activation function to ensure the length of vector in between 0-1 where short vectors get shrunk to almost zero length and long vectors get shrunk to a length slightly below 1. 
$$v_{j} = \dfrac{||s_j||^2}{1 + ||s_j||^2} \dfrac{s_j}{||s_j||}$$ <br>
where $$v_j$$ is the output vector of capsule $$j$$ and $$s_j$$ is the total input from capsule $$j$$. <br>
- Note <br>
The derivative of $$||s_j||$$ is undefined when $$||s_j||=0$$. During training if a vector is zero, the gradients will be nan. To avoid this situation we add a tiny epsilon with squared norm then apply square root. <br>
$$||s_j||  \approx  \sqrt{\sum\limits_{i}{{s_{i}}^{2}} + \epsilon}$$

The second part of sqash function $$\dfrac{s_j}{||s_j||}$$ is a unit vector means its length is 1 and the first part $$\dfrac{||s_j||^2}{1 + ||s_j||^2}$$ is a scalar, we scale the unit vector with this scalar to ensure long vectors length is close to 1 and short length is close to zero. 

Now lets experiment this effect with some dummy numbers,


```python
x = np.linspace(1, 5, 1000)
y = x / (1 + x)
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```


![squash]({{ '/assets/images/posts/2018-11-13-demystify-capsule-network-using-pytorch/squash.png' | relative_url }})


We can see for large value of $$x$$, $$y$$ gets close to 1 and for small value gets close to 0. Here, we assume the values $$x$$ are the dot product/squared norm of any vector.


We compute the squash function for all 8 dimensional capsule vector.


```python
def safeNorm(tensor, dim, epsilon=1e-7, keepdim=True):
    squared_norm = tensor.pow(2).sum(dim=dim, keepdim=keepdim)
    safe_norm = torch.sqrt(squared_norm + epsilon)
    return safe_norm, squared_norm
```


```python
def squash(tensor, dim, keepdim=True):
    safe_norm, squared_norm = safeNorm(tensor, dim=dim, keepdim=keepdim)
    squash_factor = squared_norm / (1. + squared_norm)
    unit_vector = tensor / safe_norm
    return squash_factor * unit_vector
```


```python
primary_caps_vec = squash(primary_capsule_out, dim=-1)
primary_caps_vec.size()
```




    torch.Size([5, 1152, 8])



## Digit Capsule layer
This final layer of Encoder has one 16 dimensional capsule for each digit class and each of these capsules receives input from all the capsules in Primary Capsule layer.

Every capsule in the Primary layer tries to predict the output of every capsule in Digit layer. Output of primary capsules only send to those capsule in Digit capsule, if primary capsules prediction agrees with the ouput of digit capsule. We take this decision by 'Routing By Agreement'. Digit capsules will get only the appropriate output from primary capsules and more accurately determine the spatial information. We route only between primary and digit capsule because first convolution layer encode lower level features, there is no spatial information in its space to agree on. 
>A lower-level capsule prefers to send its output to higher level capsules whose activity vectors have a big scalar product with the prediction coming from the lower-level capsule.

Using the output vector of primary capsule we will predict the output vector of digit capsule, both layer are fully connected. To implement this, we first need a weight matrix $$W_{ij}$$ for each pair of capsule in primary and digit layer. Since the output vector of primary capsule is 8 dimensional and digit capsule is 16 dimensional, $$W_{ij}$$ will be shape of (16, 8) for each pair. We get the 'prediction vectors' $$\hat{u}_{j|i}$$ for each pair of capsules (1152 in primary and 10 in digit) by multiplying primary capsule output $$u_{i}$$ and weight matrix $$W_{ij}$$.
$$\hat{u}_{j|i} = W_{ij}u_{i}$$ <br>
So, for all capsule pair the shape of $$w_{ij}$$ will be [batch_size, 1152, 10, 16, 8] and we need to make 10 copy of all 1152 primary capsules as we have 10 digit capsule so $$u_i$$ will be [batch_size, 1152, 10, 8]. To multiply these matrix we need to expand a dimension of $$u_i$$. Final shape of $$u_i$$ will be [batch_size, 1152, 10, 8, 1]. 

Now create trainable weight matrix of size [batch_size, 1152, 10, 16, 8] using standard normal distribution.


```python
digit_num_capsule = 10
digit_capsule_dim = 16
W = nn.Parameter(torch.randn([primary_num_capsule, digit_num_capsule, digit_capsule_dim, primary_capsule_dim]))
print(W.size())
W = torch.stack([W] * batch_size, dim=0)
W.size()
```

    torch.Size([1152, 10, 16, 8])
    
    torch.Size([5, 1152, 10, 16, 8])



Now we will make 10 copies of primary capsule output vectors then expand a dimension.


```python
u = torch.stack([primary_caps_vec] * digit_num_capsule, dim=2)
print(u.size())
u = u.unsqueeze(-1)
u.size()
```

    torch.Size([5, 1152, 10, 8])
    
    torch.Size([5, 1152, 10, 8, 1])




```python
u_hat = torch.matmul(W, u)
u_hat.size()
```




    torch.Size([5, 1152, 10, 16, 1])



### Routing by Agreement
![Routing algorithm]({{ '/assets/images/posts/2018-11-13-demystify-capsule-network-using-pytorch/routing-algo.png' | relative_url }})

First we need to define initial routing logits $$b_{ij}$$ for each capsule pair, which are the log prior probabilities that determine primary capsule i should be coupled to digit capsule j. This log priors are initially zero because we don't know which primary capsule should be coupled with which digit capsule initially. This log priors can be learned discriminatively at the same time as all the other weights.


```python
b_ij = Variable(torch.zeros(batch_size, primary_num_capsule, digit_num_capsule, 1, 1))
## Two additional dimension added to make easy matrix multiplication.
b_ij.size()
```




    torch.Size([5, 1152, 10, 1, 1])



Then we softmax this log prior along the digit capsule dimension to get the probability for each primary and digit capsule pair, which is the routing weights $$c_{ij}$$.


```python
c_ij = F.softmax(b_ij, dim=2)
c_ij.size()
```




    torch.Size([5, 1152, 10, 1, 1])



Now compute the weighted sum of all the 'predicted output vectors' $$\hat{u}_{j|i}$$ for each digit capsule.
$$s_j = \sum{c_{ij}\hat{u}_{j|i}}$$ 


```python
s_j = (c_ij * u_hat)#.sum(dim=1, keepdim=True)
print(s_j.size())
s_j = s_j.sum(dim=1, keepdim=True)
s_j.size()
```

    torch.Size([5, 1152, 10, 16, 1])

    torch.Size([5, 1, 10, 16, 1])



To perform elementwise matrix multiplication it requires requires 'routing weights' and 'prediction vectors' to have the same rank, which is why we added two extra dimensions of size 1 to routing_weights, earlier. 
The shape of 'routing weights' is (batch_size, 1152, 10, 1, 1) while the shape of 'prediction vectors' is (batch_size, 1152, 10, 16, 1). Since they don't match on the fourth dimension (1 vs 16), pytorch will automatically broadcasts the 'routing weights' 16 times along that dimension.

The output of digit capsule $$s_j$$ might give vectors larger than 1, so we need to squash these 16 dimensional vector by $$v_j = squash(s_j)$$. 


```python
v_j = squash(s_j, dim=-2)
v_j.size()
```




    torch.Size([5, 1, 10, 16, 1])



Now we need to determine which primary capsule agrees with which digit capsule. We just simply compute it by the scalar product of each instance between predicted vector by primary capsule $$\hat{u}_{i|j}$$ and digit capsules output $$v_j$$.
$$a_{ij} = v_j .\hat{u}_{j|i}$$


```python
print(u_hat.size())
print(v_j.size())
```

    torch.Size([5, 1152, 10, 16, 1])
    torch.Size([5, 1, 10, 16, 1])
    

Since second dimension does not match we need to make 1152 copies of digit capsule prediction.


```python
torch.cat([v_j] * primary_num_capsule, dim=1).size()
```




    torch.Size([5, 1152, 10, 16, 1])



As we are cumputing the agreement of 16 dimensional vector, to multiply these vector we need to transpose last two dimension of any of these vector.


```python
a_ij = torch.matmul(v_j.transpose(3, 4), u_hat)
a_ij.size()
```




    torch.Size([5, 1152, 10, 1, 1])



We can now update the raw routing weights $$b_{i,j}$$ by simply adding the agreement $$a_{ij}$$


```python
b_ij = b_ij + a_ij
b_ij.size()
```




    torch.Size([5, 1152, 10, 1, 1])



According to the paper, we need to iterate step 4 to 7 three times.

## Margin loss
The paper uses a special margin loss to discriminate different digits. <br>
$$ L_k = T_k \max(0, m^{+} - ||v_k||)^2 + \lambda (1 - T_k) \max(0, ||v_k|| - m^{-})^2$$<br>

* $$T_k$$ is equal to 1 if the digit of class $$k$$ is present, otherwise 0.
* In the paper, $$m^{+} = 0.9$$, $$m^{-} = 0.1$$ and $$\lambda = 0.5$$.
* Loss will be 0 if the correct DigitCap predicts the correct label with greater than 0.9 probability, and it will be non-zero if the probability is less than 0.9.
* Loss will be zero if the mismatching DigitCap predicts an incorrect label with probability less than 0.1 and non-zero if it predicts an incorrect label with probability more than 0.1.

We scale down the loss for incorrect classes so that these majority classes don't dominate the learning procedure.


```python
m_plus = 0.9
m_minus = 0.1
lambda_ = 0.5
```

To get $$T_k$$ for every instance and every class we need to transform the labels into one-hot encoding.


```python
print('raw labels - ', labels)
T = torch.from_numpy(labels)
T = torch.eye(10).index_select(dim=0, index=T)
print('One-hot labels - ', T)
```

    raw labels -  [8 4 2 6 1]
    One-hot labels -  tensor([[0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]])
    

The 16 dim output vectors are in the second to last dimension, so let's use the safeNorm() function with dim=-2:


```python
v_k, _ = safeNorm(v_j, dim=-2, keepdim=True)
v_k.size()
```




    torch.Size([5, 1, 10, 1, 1])



Now let's compute $$\max(0, m^{+} - \|\mathbf{v}_k\|)^2$$, and reshape the result to get a simple matrix of shape (batch size, 10):


```python
correct_loss = F.relu(m_plus - v_k).pow(2).view(5, -1)
correct_loss.size()
```




    torch.Size([5, 10])



Next let's compute $$\max(0, \|\mathbf{v}_k\| - m^{-})^2$$ and reshape it:


```python
incorrect_loss = F.relu(v_k - m_minus).pow(2).view(5, -1)
incorrect_loss.size()
```




    torch.Size([5, 10])



Now to compute total margin loss for each instance and each digit.


```python
margin_loss = T*correct_loss + lambda_*(1 - T)*incorrect_loss 
margin_loss = margin_loss.sum(dim=1).mean()
margin_loss
```




    tensor(2.0924, grad_fn=<MeanBackward1>)



## Reconstruction
In paper an additional reconstruction loss was used to encourage the digit capsules to encode the instantiation parameters of the input digit. During training, instead of using all the 16 dimensinal digit capsule they only use the capsule corresponds to the target digit and masked out other capsules to reconstruct the input image. These 16 dimensional digit capsules are feed into a decoder consisting of 3 fully connected layers and minimize the sum of squared differences between the outputs of the logistic units and the pixel intensities. Reconstruction loss was scale down by 0.0005 so that it does not dominate the margine loss during training. This loss works as regularization to reduce the risk of overfitting.<br>
![Reconstruction]({{ '/assets/images/posts/2018-11-13-demystify-capsule-network-using-pytorch/reconstruction.png' | relative_url }})



```python
v_j = v_j.squeeze(1) # eleminate additional dimension
v_j.size()
```




    torch.Size([5, 10, 16, 1])



Now we have to calculate the length of every 16 dimensional digit capsule


```python
classes = torch.sqrt((v_j ** 2).sum(dim=2, keepdim=False))
print(classes.size())
classes[:1]
```

    torch.Size([5, 10, 1])

    tensor([[[0.8712],
             [0.8848],
             [0.7797],
             [0.7473],
             [0.6140],
             [0.8229],
             [0.8561],
             [0.8210],
             [0.6922],
             [0.7880]]], grad_fn=<SliceBackward>)



Now we need to find the correct class digit capsule predicted from the length of capsules.


```python
_, max_length_indices = classes.max(dim=1)
_, max_length_indices, 
```




    (tensor([[0.8848],
             [0.8567],
             [0.8594],
             [0.8518],
             [0.8448]], grad_fn=<MaxBackward0>), tensor([[1],
             [6],
             [6],
             [0],
             [9]]))



Lets construct a one-hot matrix containing the digit capsule predicted class, we will use this matrix to musk out the other activity vector of digit capsule.


```python
masked = Variable(torch.eye(10))
masked.size()
```




    torch.Size([10, 10])




```python
masked = masked.index_select(dim=0, index=max_length_indices.squeeze(1).data)
print(masked.shape)
masked
```

    torch.Size([5, 10])
    tensor([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])



To musk out the incorrect activity vectors of digit capsule we need to multiply the capsules with this one-hot vector. To satisfy the multiplication, dimension of the activity vector and the one-hot vector must be same.


```python
masked[:, :, None, None].size()
```




    torch.Size([5, 10, 1, 1])




```python
decoder_input = v_j * masked[:, :, None, None]
decoder_input.size()
```




    torch.Size([5, 10, 16, 1])



Now build the decoder layer with two fully-connected hidden layer followed by Relu activation and output layer followed by Sigmoid activation using the parameter mention in the paper.


```python
decoder_layer = nn.Sequential(nn.Linear(10 * 16, 512), 
                              nn.ReLU(inplace=True),
                              nn.Linear(512, 1024),
                              nn.ReLU(inplace=True),
                              nn.Linear(1024, 28 * 28),
                              nn.Sigmoid())
```

To feed the decoder input into the decoder layer we need to flat the input for each batch.


```python
decoder_input = decoder_input.view(batch_size, -1)
decoder_input.size()
```




    torch.Size([5, 160])




```python
decoder_output = decoder_layer(decoder_input)
decoder_output.size()
```




    torch.Size([5, 784])



This decoder output is the reconstructed images of the input images. To plot the output image we need to resize this output vectors into 28x28 pixel like mnist.

## Reconstruction Loss
Now let's compute the reconstruction loss. It is just the sum squared difference between the input image and the reconstructed image.


```python
mse = nn.MSELoss()
reconstruction_loss = mse(decoder_output, input_images.view(batch_size, -1))
reconstruction_loss
```




    tensor(0.0834, grad_fn=<MseLossBackward>)



## Final Loss
The final loss is the sum of the margin loss and the reconstruction loss (scaled down by a factor of 0.0005 to ensure the margin loss dominates training)


```python
alpha = 0.0005
final_loss = margin_loss + (alpha * reconstruction_loss)
final_loss
```




    tensor(2.0925, grad_fn=<ThAddBackward>)



## Reference
1. [Dynamic Routing Between Capsules](https://arxiv.org/pdf/1710.09829.pdf) by Sara Sabour, Nicholas Frosst and Geoffrey E. Hinton<br>
2. [Understanding Hinton’s Capsule Networks](https://medium.com/ai%C2%B3-theory-practice-business/understanding-hintons-capsule-networks-part-i-intuition-b4b559d1159b) by Max Pechyonkin<br>
3. [Capsule Networks (CapsNets) – Tutorial](https://www.youtube.com/watch?v=pPN8d0E3900&t=10s) by Aurélien Géron
