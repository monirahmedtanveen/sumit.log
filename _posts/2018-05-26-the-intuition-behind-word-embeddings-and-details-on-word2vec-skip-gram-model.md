---
layout: post
comments: false
title: "The Intuition behind Word Embeddings And Details On Word2vec Skip-gram Model"
date: 2018-05-26 10:00:00
tags: nlp word2vec skip-gram review
---

>  Word2Vec is a group of models (skip-gram and continuous bag of words) that tries to represent each word in a large text as a vector in a space of N dimensions to preserve the semantic and syntactic relationships of words. Currently, word2vec motheds are extensively used in Natural Language Processing.


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}


This tutorial covers the essentials of word embeddings in Natural Language Processing and detail explanation of word2vec skip-gram model. You can find the source code of skip-gram model 
<a href="https://github.com/sakhawathsumit/word2vec-skip-gram"><font color='black'>here</font></a>.
<font color='black'><h2>word embeddings</h2></font>
At this moment, Word Embeddings are the state of the art in Natural Language Processing where the words are represented as vectors of continuous real numbers.<br>
$$W('cat')=(0.1,0.5,-0.2,...)$$<br>
$$W('dog')=(0.2,0.4,-0.2,...)$$<br>
The idea of Word Embeddings originally introduced by <a href="http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf"><font color='black'>Bengio et al.(2003)</font></a> in the paper "A Neural Probabilistic Language Model". This paper generally focused on reducing the high dimentionality of word representations in context by "learning a distributed representation for words" and has a great insight about the efficacy of word embeddings.<br>
Generally, a feed-forward neural network takes the words from a vocabulary as input and embeds them as vectors, then optimized through a back-propagation algorithm. The weights of the first layer is used as word embeddings which is often refered as Embedding Layer. After the embeddings, semantically similar words have similar vectors.<br>

![Word Embedding]({{ '/assets/images/posts/2018-05-26-the-intuition-behind-word-embeddings-and-details-on-word2vec-skip-gram-model/word_embed.jpeg' = 500x400 | relative_url }})
<center>Fig 1. Word embeddings from <a href="http://www.iro.umontreal.ca/~lisa/pointeurs/turian-wordrepresentations-acl10.pdf"><font color='black'>Turian et al.(2010)</font></a></center>

<br>In the figure 1, neumerical values are closer to each other as they have semantically similar meaning. And same proparties goes for others.

<font color='black'><h2>Language Model</h2></font>
Word embeddings are the most significant part of language models. Language model is a probability distribution over a sequences of words. The quality of language models is measured based on their ability to learn a probability distribution over a set of vocabulary in V.<br>
Language models generally try to compute the probability of a word $${ w }_{ i }$$ given its previous $$n-1$$ words. i.e. $$p({ w }_{ i }|{ w }_{ i-1 }{ ,w }_{ i-2,..., }{ w }_{ i-n+1 })$$. By applying the markov chain rule, we can approximate the product of a sentence by the product of the probabilities of each words given its previous $$n$$ words:
$$p({ w }_{ 1 },...,{ w }_{ n })=\prod _{ i }^{  }{ p(w_{ i }|{ w }_{ i-1 },...,{ w }_{ i-n+1 }) } $$<br>
<b>n-gram model</b> is a type of probabilistic language model which uses the previous $$n-1$$ words in a text corpus to predict the next word. In an n-gram model, the probability $$P({ w }_{ 1 },...,{ { w }_{ n } })$$ of observing the sentence $${ w }_{ 1 },...,{ { w }_{ n } }$$ is approximated as:
$${ P(w }_{ 1 },...,{ { w }_{ n } })=\prod _{ i=1 }^{ n }{ P({ w }_{ i }|{ w }_{ 1 },...,{ w }_{ i-1 })\approx \prod _{ i=1 }^{ n }{ P({ w }_{ i }|{ w }_{ i-(n-1) },...,{ w }_{ i-1 }) }  } $$<br>
Here, it is assumed that the property of observing the $${ i }^{ th }$$ word $${ w }_{ i }$$ in the context history of the preceding $$i-1$$ words can be approximated by the probability of observing it in the shortened context history of the preceding $$n-1$$ words($${ n }^{ th }$$ order markov property).<br>
In n-gram models, we can calculate a word's probability based on frequencies of its constituent n-grams:
$$P({ w }_{ i }|{ w }_{ i-(n-1) },...,{ w }_{ i-1 })=\frac { count({ w }_{ i-(n-1) },...,{ w }_{ i-1 },{ w }_{ i }) }{ count({ w }_{ i-(n-1) },...,{ w }_{ i-1 }) } $$
The words <b>bigram</b> and <b>trigram</b> language model denote n-gram model language models with $$n=2$$ and $$n=3$$, respectively. While $$n=5$$ together with <a href="https://en.wikipedia.org/wiki/Kneser%E2%80%93Ney_smoothing"><font color='black'>Kneser-ney smoothing</font></a> leads to smoothed 5-gram model that have been found to be a strong baseline for language model(<a href="https://web.stanford.edu/class/cs124/lec/languagemodeling.pdf"><font color='black'>see more</font></a> from Stanford).<br>
In neural networks, we can achieve the same objective using the Softmax layer:
$$P({ w }_{ i }|{ w }_{ i-(n-1) },...,{ w }_{ i-1 })=\frac { exp({ h }^{ \top  }{ v }_{ { w }_{ i } }^{ ' }) }{ \sum _{ { w }_{ j }\in V }^{  }{ exp({ h }^{ \top  }{ v }_{ { w }_{ j } }^{ ' } } ) } $$
The inner product $${ h }^{ \top  }{ v }_{ { w }_{ i } }^{ ' }$$ computes the unnormalized log-probability of word $${ w }_{ i }$$ which we normalize by the sum of log-probabilities of all words in $$V.h$$. <br>
<b>The neural probabilistic language model</b> by Benjio et al.(2003) consists of one-hidden layer feed-forword neural network that predicts next word in a text corpus.

![Language Model]({{ '/assets/images/posts/2018-05-26-the-intuition-behind-word-embeddings-and-details-on-word2vec-skip-gram-model/bengio_nural_language_model.png' =500x350 | relative_url }})
<center>Fig 2. Neural language model from <a href="http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf"><font color='black'>Benjio et al.(2003)</font></a></center>

Their model maximizes the probability distribution as described above. The neural probabilistic language model's objective function:
$${ J }_{ \theta  }=\frac { 1 }{ N } \sum _{ i=1 }^{ N }{ log\quad f({ w }_{ i },{ w }_{ i-1 },...,{ w }_{ i-n+1 } } )$$
$$f({ w }_{ i },{ w }_{ i-1 },...,{ w }_{ i-n+1 })$$ is the output of the model. i.e. the probability $$p({ w }_{ i }|{ w }_{ i-1 },...,{ w }_{ i-n+1 })$$ as computed by the softmax, where $$n$$ is the number of previous words feed into the model.<br><br>
Their model is better at fighting the <b>Data Sparsity problem</b> (most possible word sequences will not be observed in training) for large $$n$$ where it is a major problem in building language model. The reason is, it can potentially generalize to contexts that have not been seen in training set.<br><br>
Here is an example for p(' eating '|'the','cat','is'), imagine that we want to evaluate the probability of the word 'eating' after seeing the words 'the', 'cat', 'is'.<br>
Now suppose, in 4-gram ['the','cat','is','eating'] is not in training corpus, but ['the', 'dog', 'is', 'eating'] is in the training corpus. If the word embeddings of 'cat' and 'dog' has similar vector and if the neural network has learned to predict good probability for word 'eating' after seeing 'the', 'dog', 'is' then the neural network will be able to predict similar probability in the context where instead of dog we have cat. This is because the neural network may have seen 'cat' and 'dog' in a similar context. Like, ['the','cat','was','sleeping'] and ['the', 'dog', 'was', 'sleeping'].

<font color='black'><h2>Word2Vec Skip-Gram Model</h2></font>
Now let's look into word2vec. It's a group of related models that are used to learn vector representation of words from a large text corpus, called 'word embeddings'. Word2vec was created by <a href="http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf"><font color='black'>Mikolov et al.(2013)</font></a> at Google. It comes in two flavors, the Continuous Bag-of-Words (CBOW) and the Skip-Gram model, this models are the state of the art in word embeddings. CBOW predicts target words(e.g. 'mat') from source context words ('the cat sits on the'), while the skip-gram does the inverse and predicts source context-words from the target words. CBOW treats an entire context as one observation and comparatively works well for smaller datasets. However, skip-gram model treats each context-target pair as a new observation and works better for larger datasets. Here I am gonna cover skip-gram model's neural network architecture and implimentation.

<font color='black'><h4>Skip-Gram Model</h4></font>
The neural network architechture of skip-gram model is quite simple. We are going to train a simple neural network with a single hidden layer. But, here the actual goal is to learn the weights of the hidden layer and these weights are the 'word vectors', which is our main agenda.<br>
For a specific word in a sentence, we look the words around it in a fixed "window size" and pick one at random. The network is going to estimate the probability for each word in the vocabulary of beign close to the choosen word in the range of window size. Suggested window size is $$5$$, the network will look $$5$$ words behind and $$5$$ words ahead in total $$10$$.<br>
The output probabilities of words which much likely to appear next to each other or related are higher than the words which are not appearing in similar context or related. For example, if in training session the networks sees word "United" as input, the output probabities are going to be much higher for words like "State" and "Kingdom" than for unrelated words like "Sushi" and "Tofu".<br>

![Skip-gram]({{ /assets/images/posts/2018-05-26-the-intuition-behind-word-embeddings-and-details-on-word2vec-skip-gram-model/skip-gram.png' =400x350 | relative_url }})
<center>Fig 3. Skip-gram model</center>

We'll train the neural network by feeding word pairs (input, output) from our dataset. As an example, let's consider the sentence<br>
**"the quick brown fox jumped over the lazy  dog"**<br>
First, we have form a training dataset of words, depending on their appear in a word window. using a window size of 1, we then have the dataset<br>
**([the, brown], quick), ([quick, fox], brown), ([brown, jumped], fox), ...** <br>
of **(context, target)** pairs, Recall that skip-gram inverts contexts and targets and tries to predict each context word from its target word, so the task becomes to predict 'the' and 'brown' from 'quick', 'quick' and 'fox' from 'brown' etc. Therefore out dataset becomes
**(quick, the), (quick, brown), (brown, quick), (brown, fox), ...**<br>

<font color='black'><h4>Model Details</h4></font>
We can not feed word as string to a neural network, so we have to represent the words as vectors. To do this , we first build a vocabulary of words from our training set. Let's say we have 10,000 unique words. We give an unique ID to every word and represent an input word like "ants" as a one-hot vector. This vector will have 10,000 components (one for every word in our vocabulary) and we’ll place a “1” in the position corresponding to the ID of word “ants”, and 0s in all of the other positions.<br>
The output of the network is a single vector (also with 10,000 components) containing, for every word in our vocabulary, the probability that a randomly selected nearby word is that vocabulary word. It is actually a probability distribution (i.e., a bunch of floating point values, not a one-hot vector).

![Neural Network Architecture]({{ /assets/images/posts/2018-05-26-the-intuition-behind-word-embeddings-and-details-on-word2vec-skip-gram-model/nn_architechture.png' =500x350 | relative_url }})
<center>Fig 4. Neural Network Architecture</center>

There is no activation function on the hidden layer neurons, but the output neurons use softmax. We’ll come back to this later.<br>

<font color='black'><h4>The Hidden Layer</h4></font>
For our example, we’re going to say that we’re learning word vectors with 300 features. So the hidden layer is going to be represented by a weight matrix with 10,000 rows (one for every word in our vocabulary) and 300 columns (one for every hidden neuron).<br>
If you look at the rows of this weight matrix, these are actually what will be our word vectors!

![Hidden Layer]({{ /assets/images/posts/2018-05-26-the-intuition-behind-word-embeddings-and-details-on-word2vec-skip-gram-model/hidden_layer.png' =500x350 | relative_url }})

So the end goal of all of this is really just to learn this hidden layer weight matrix – the output layer we’ll just toss when we’re done!<br>
Let’s get back, though, to working through the definition of this model that we’re going to train.<br>
Now, you might be asking yourself–“That one-hot vector is almost all zeros… what’s the effect of that?” If you multiply a 1 x 10,000 one-hot vector by a 10,000 x 300 matrix, it will effectively just select the matrix row corresponding to the “1”. Here’s a small example to give you a visual.

![Matrix Multiplication]({{ /assets/images/posts/2018-05-26-the-intuition-behind-word-embeddings-and-details-on-word2vec-skip-gram-model/matrix_multiplication.png' =500x250 | relative_url }})

This means that the hidden layer of this model is really just operating as a lookup table. The output of the hidden layer is just the “word vector” for the input word.

<font color='black'><h4>The Output Layer</h4></font>
The $$1 x 300$$ word vector for “ants” then gets feed to the output layer. The output layer is a softmax regression classifier. There’s an in-depth tutorial on Softmax Regression <a href="http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/"><font color='black'>here</font></a>, but the gist of it is that each output neuron (one per word in our vocabulary!) will produce an output between 0 and 1, and the sum of all these output values will add up to 1.<br>
Specifically, each output neuron has a weight vector which it multiplies against the word vector from the hidden layer, then it applies the function $$exp(x)$$ to the result. Finally, in order to get the outputs to sum up to 1, we divide this result by the sum of the results from all 10,000 output nodes.<br>
Here’s an illustration of calculating the output of the output neuron for the word "car".

![Output Layer]({{ /assets/images/posts/2018-05-26-the-intuition-behind-word-embeddings-and-details-on-word2vec-skip-gram-model/output_layer.png' =500x250 | relative_url }})

<font color='black'><h2>References</h2></font>
1. Benjio et al.(2003). A Neural Probabilistic Language Model. Journal of Machine Learning Research (3)1137–1155
2. Mikolov et al.(2013). Distributed Representations of Words and Phrases
and their Compositionality. arXiv:1310.4546
3. Turian et al. (2010). Word representations: A simple and general method for semi-supervised learning. 
4. <a href="http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/"><font color='black'>http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model</font></a>
