---
layout: post
comments: false
title: "A Simple Introduction To Softmax Function"
date: 2018-04-08 00:15:06
tags: softmax review
---

>  The softmax function is often used in the final layer of a neural network-based classifier. It calculates a probability distribution for multiclass problem. Neural Network uses this probability distribution to predict output class. 


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}


The softmax function also refered as "normalized exponential function" takes an N-dimensional vector of arbitrary real values and generates another N-dimensional vecotor with real values in the range (0, 1) that add up to 1.0. <br>
It maps $S(x) :\mathbb{R}^N \rightarrow \mathbb{R}^N$ :

$S(x) : \begin{bmatrix}
    x_{1}\\
    x_{2}\\
    x_{3}\\
    \vdots \\
    x_{N}
\end{bmatrix} \rightarrow \begin{bmatrix}
    S_{1}\\
    S_{2}\\
    S_{3}\\
    \vdots \\
    S_{N}
\end{bmatrix}$ <br>
For a particular element in the vector the formula is : $S_j = \frac { { e }^{ x\_ j } }{ \sum _{ i=1 }^{ N }{ { e }^{ x\_ i } }  }$  &nbsp;&nbsp;  for $i = 1, ..., N$ <br>
The result of exponent of any real number is always $\ge 0$. Since, the numerator appears in the denominator and summed up with other positive numbers so, $S_j < 1$ and in the range (0, 1).


```python
import numpy as np

def softmax(inputs):
    """
    Calculate softmax for a given input vector.
    """
    return np.exp(inputs) / float(sum(np.exp(inputs)))

inputs = [2, 1, 4, 7]
softmax(inputs)
```




    array([ 0.00636253,  0.00234065,  0.04701312,  0.9442837 ])



Here, softmax function pushes the larger values close to 1 and smaller values close to 0 and their summation is 1.0 which preserved the property. As a result, largest value of the vector remains largest in the output vector. Softmax function is a "soft" version of Hardmax or Standard Maximum function. Harmax function selects the maximum value of the vector where softmax function generates a probability distribution from the input vector.

Exponential function of softmax can overshoot modest size inputs which results to Overflow of precision points.


```python
softmax([200, 5000, 1500])
```

    /home/sumit/tensorflowGPU/lib/python3.5/site-packages/ipykernel_launcher.py:7: RuntimeWarning: overflow encountered in exp
      import sys
    /home/sumit/tensorflowGPU/lib/python3.5/site-packages/ipykernel_launcher.py:7: RuntimeWarning: invalid value encountered in true_divide
      import sys
    




    array([  0.,  nan,  nan])



To avoiding this Overflow problem we need to normalize the inputs so that it can not be too large or too small. For that, we need to use an arbitrary constant C. <br>
$ S_j = \frac { C{ e }^{ x\_ j } }{ \sum _{ i=1 }^{ N }{ C{ e }^{ x\_ i } }  }$ <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$=\frac { { e }^{ x\_ j + \log { (C) }  } }{ \sum _{ i=1 }^{ N }{ e^{ x\_ i + \log { (C) }  } }  } $ <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$=\frac { { e }^{ x\_ j + D } }{ \sum _{ i=1 }^{ N }{ e^{ x\_ i + D } }  } $ &nbsp;&nbsp; Since C is a Constant we can replace it with D. <br>
Here, D also an arbitrary constant and to avoid overflow we can choose D as maximum of inputs and negate it. <br>
$D=-max(x_1,x_2,x_3,...,x_N)$ <br>
If the interdistance of the inputs is not large it shift the inputs close to 0. All inputs become negative except the maximum $x_j$ becomes 1. Expenents saturates the large negative values to zeros.


```python
def softmax_updated(inputs):
    new_inputs = inputs - np.max(inputs)
    return np.exp(new_inputs) / float(sum(np.exp(new_inputs)))
softmax_updated([200, 5000, 1500])
```




    array([ 0.,  1.,  0.])



As softmax function calculates the probablity distribution of any vector, it is generally used in various multiclass classification methods in the areas of machine learning, deep learning and data science. Softmax calculates the probabilities of each target class over all possible target classes. The calculated probabilities helps to determine the correct target class for a given input set.<br>
Softmax function most usage:
* Logistic regression.
* Artificial neural networks.
