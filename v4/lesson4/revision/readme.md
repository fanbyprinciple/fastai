# Chapter 4

the familiar 3:
![](rthree.png)

The mean of all seven tensors:
![](rmean.png)

why to use tensors than normal array because they are fast

Numpy and pytorch tensor vaies that that pytorch tenorshoud have same basic type for all basic arrays.

Pytorch structures aremade for GPU, and can automatically find derivatives.

metric is for calculating how good a model is

broadcasting does mean using extra memory

wrong classification by mean model:
![](rwrong.png)

Illustrative function to use SGD
![](rillus.png)

gradient desent after 10020 steps
![](rsgd.png)

The key difference is that the metric is to drive human understanding and the loss is to drive automated learning. To drive automated learning, the loss must be a function that has a meaningful derivative. 

since we can;t apply the loss calculation and update on a single set nor on a big dataset, we take mini batches.

The validation accuracy drop then increases sure is a mystery.
![rmystery](rmystery.png)

