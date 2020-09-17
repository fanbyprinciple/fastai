# MNIST basics

Training a digit classifier

![](three_top.png)


https://colab.research.google.com/drive/1MybjxDCbLLJCVQR5Ahu552VqDRJosHFG#scrollTo=bptSpUapoHgH

https://colab.research.google.com/github/fastai/fastbook/blob/master/04_mnist_basics.ipynb#scrollTo=Y3vljyfChrB_

### Aim of the chapter
1. to understand the roles of array and tensors in broadcasting and using them expressively

2. We will learn stohastic gradient descent by updating the weights automatically

3. We will also describe the math the neural network is doinf

4. Role of mini batches


There's one guaranteed way to fail, and that's to stop trying. We've seen that the only consistent trait amongst every fast.ai student that's gone on to be a world-class practitioner is that they are all very tenacious.


Here we are using the python imaging library (PIL) which is the widely used Python package for opening, manipulating and viewing images.

![](new_3.png)

to see the image we need to convert it into a Numpy array or a pytorch tensor

some operations in pytorch we need to case to integer and float types. Since we be needing this later, we will also cast our stacked tensor to float

Generally when images are floats the picel values are expected to be between 0 and also devide by 255
Since we'll be needing this later, we'll also cast our stacked tensor to float now. Casting in PyTorch is as simple as typing the name of the type you wish to cast to, and treating it as a method.

Rank is the number of axes in a tensor while shape is the size of each axis of tensor

Here mse stands for mean squared error, and l1 refers to the standard mathematical jargon for mean absolute value (in math it's called the L1 norm).
: Intuitively, the difference between L1 norm and mean squared error (MSE) is that the latter will penalize bigger mistakes more heavily than the former (and be more lenient with small mistakes).

For our simple avergaing model the loss:
![](loss_average.png)

Next step :
 NUmpy and pytorch array
