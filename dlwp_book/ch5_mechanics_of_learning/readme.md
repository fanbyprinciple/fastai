# The mechanics of Learning

## Timeless lessons in modelling

In this book, we’re interested in models that are not engineered for solving a specific narrow task, but that can be automatically adapted to specialize themselves for
any one of many similar tasks using input and output pairs—in other words, general
models trained on data relevant to the specific task at hand. 

## Learning is a parameter estimation

In order to optimize the parameter of the model—its weights—the change in
the error following a unit change in weights (that is, the gradient of the error with
respect to the parameters) is computed using the chain rule for the derivative of a
composite function (backward pass). The value of the weights is then updated in the
direction that leads to a decrease in the error. The procedure is repeated until the
error, evaluated on unseen data, falls below an acceptable level

## simple prediction model

![](prediction.png)

After modelling :

![](prediction_f.png)


## Pytorch autograd

we just saw a simple example of backpropagation: we computed the gradient of a composition of functions—the model and the loss—with
respect to their innermost parameters (w and b) by propagating derivatives backward
using the chain rule

## Computing gradient automatically

This is when PyTorch tensors come to the rescue, with a PyTorch component called
autograd

![](autograd.png)

## optimisers ala carte

https://www.kaggle.com/fanbyprinciple/simplest-model-parameter-estimation/edit

page 158

