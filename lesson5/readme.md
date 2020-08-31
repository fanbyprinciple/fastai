# Foundations of neural networks
https://www.youtube.com/watch?v=CJKnDu2dxOE&vl=en

1 hour 7 min

https://github.com/hiromis/notes/blob/master/Lesson5.md


There are two kinfs of layers

1. Parameter + weights layer
2. activations layer
its an element wise function like relu

Followed by gradient descent - this is true in back propagation

## Fine tuning

In imagenet the target vetor is 1000. 
So what we do in fine tuning is unfreeze the pretrained layers and use discriminative learning rates for handling the pretrained and our own data.

when we say slice(1e-3) it means that top layers get lr of 1e-3 and other layer get the lr /3

Affine function is something very close to deep learning

### Embedding

multiplying by a one hot encoded matrix is identical to doing an array lookup. Therefore we should always do the array lookup version, and therefore we have a specific way of saying I want to do a matrix multiplication by a one hot encoded matrix without ever actually creating it. I'm just instead going to pass in a bunch of integers and pretend they're one not encoded. And that is called an embedding.

 embedding means look something up in an array. But it's interesting to know that looking something up in an array is mathematically identical to doing a matrix product by a one hot encoded matrix. And therefore, an embedding fits very nicely in our standard model of our neural networks work.

Now suddenly it's as if we have another whole kind of layer. It's a kind of layer where we get to look things up in an array. But we actually didn't do anything special. We just added this computational shortcut - this thing called an embedding which is simply a fast memory efficient way of multiplying by hot encoded matrix.

So this is really important. Because when you hear people say embedding, you need to replace it in your head with "an array lookup" which we know is mathematically identical to matrix multiply by a one hot encoded matrix.

### latent features
 the only way that this gradient descent could possibly come up with a good answer is if it figures out what the aspects of movie taste are and the corresponding features of movies are. So those underlying kind of features that appear that are called latent factors or latent features. 

### bias
Neural network can work without bias however they are not as efficient
 it's better because it's giving both more flexibility and it also just makes sense semantically that you need to be able to say whether I'd like the movie is not just about the combination of what actors it has, whether it's dialogue-driven, and how much action is in it but just is it a good movie or am i somebody who rates movies highly.

 the first argument to fit_one_cycle or fit is number of epochs. In other words, an epoch is looking at every input once. If you do 10 epochs, you're looking at every input ten times.

## Colab notebook discussion


In pytorch all classes are essentially functions

when you call a class it actually calls the forward method of the class defined inside the class

