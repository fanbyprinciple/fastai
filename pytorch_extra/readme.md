# Pytorch programs

# pytorch images captcha 

https://www.kaggle.com/fanbyprinciple/pytorch-captcha/edit?rvi=1

![](captcha_output.png)

# Using style transfer
https://www.kaggle.com/fanbyprinciple/style-transfer-using-pytorch/edit

at low epochs :
![](style_transfer1.png)

After 200 epochs :
![](style_transfer2.png)

# Variational autoencoder
https://www.kaggle.com/fanbyprinciple/creating-variational-autoencoders-in-pytorch/edit

taken from : https://www.youtube.com/watch?v=zp8clK9yCro

![](variational_autoencoder.png)

# Creating a chat application
https://www.kaggle.com/fanbyprinciple/creating-a-chatbot-with-pytorch/

![](samus_chat.gif)

# Name predictor

Official tutorial
![](rnn_name_predictor.png)

# Resnet from paper

https://www.kaggle.com/fanbyprinciple/implementing-resnet-from-paper-in-pytorch/edit

![](resnet_from_paper.png)

# Yolo

https://www.kaggle.com/fanbyprinciple/model-review-understanding-yolo-v3-with-pascal/edit

from youtube : https://www.youtube.com/watch?v=Grir6TZbc1M

46 min

![](yolo_mode.png)

# Learning Pytorch

## creating a simple neural network

![](learning_pytorch_1.png)

Somehow this model seems biased towards number 7.

https://www.kaggle.com/fanbyprinciple/learning-pytorch-1-creating-a-simple-network/edit

## creating a Convolutional neural network

![](learning_pytorch_2_custom_dataset.png)

I wonder why x expects a float while y should be long

https://www.kaggle.com/fanbyprinciple/learning-pytorch-2-creating-a-cnn/edit

I am also getting very low accuracy here.

The reason was a programmatical error while checking accuracy

![](learning_pytorch_2_correct_result.png)

## creating an RNN and GRU and LSTM

created the basis

https://www.kaggle.com/fanbyprinciple/learning-pytorch-3-coding-an-rnn-gru-lstm/edit

![](learning_pytorch_3_rnn.png)

implementing LSTM and GRU

![](learning_pytorch_3_lstm.png)

## Implementing RNN iwth time series data

https://www.youtube.com/watch?v=AvKSPZ7oyVg

https://www.kaggle.com/fanbyprinciple/learning-pytorch-4-time-sequence-with-lstm/edit

![](learning_pytorch_4_not_working.png)

https://colab.research.google.com/drive/1vy9iY5q8EbgVjgatJic-azSpMHrc6Qz5#scrollTo=Y7rt93ysFOaB

![](learning_pytorch_4_sine_wave.png)

## Implementing a bidirectional LSTM

https://www.kaggle.com/fanbyprinciple/learning-pytorch-5-creating-a-bidirectional-lstm/edit

![](learning_pytorch_5_handsign.png)

video:
https://www.youtube.com/watch?v=jGst43P-TJA&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=6

### saving model

![](learning_pytorch_5_saving_model.png)

## Transfer learning

https://www.youtube.com/watch?v=qaDe0qQZ5AQ&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=8

![](learning_pytorch_6_transfer_learning.png)

https://www.kaggle.com/fanbyprinciple/learning-pytorch-6-transfer-learning/edit

The thing to note here was when I used VGG with mnist, its images were too small. But this is a nice method to change the default models.

## Loading a dataset
https://www.youtube.com/watch?v=ZoZHd0Zm3RY&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=9

Its is showing an error right now.

![](learning_pytorch_7_error_multi_target.png)

https://www.kaggle.com/fanbyprinciple/learning-pytorch-7-custom-dataset/edit

After much code changing , it works!

![](learning_pytorch_7_custom_dataset.png)

## Loading a Text dataset

https://www.youtube.com/watch?v=9sHcLvVXsns&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=10

getting the error

1. bad caption numbers
2. cannot display proper images - handled.

![](learning_pytorch_8_error.png)

check code here :
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/custom_dataset_txt/loader_customtext.py


https://www.kaggle.com/fanbyprinciple/learning-pytorch-8-working-with-text-dataset/edit

## Augmenting dataset using torchvision

https://www.youtube.com/watch?v=Zvd276j9sZ8&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=11


Image dataset loaded

![](learning_pytorch_9_image_dataset_loaded.png)

https://www.kaggle.com/fanbyprinciple/learning-pytorch-9-data-augmentation-torchvision/edit

![](learning_pytorch_9_data_augmentation.png)

## Albumentations tutorial

https://www.youtube.com/watch?v=rAdLwKJBvPM&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=12

![](learning_pytorch_10_audrey.png)

https://www.kaggle.com/fanbyprinciple/learning-pytorch-10-albumentations-dataloader/edit

![](learning_pytorch_10_albumentation.png)

## Dealing with database imbalances

Not quite up there:

https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/Imbalanced_classes/main.py

![](learning_pytorch_11_imbalances.png)

https://www.kaggle.com/fanbyprinciple/learning-pytorch-11-dealing-with-data-imbalances/edit

It actually created too much compensation.

## Tips and tricks of training

https://www.youtube.com/watch?v=O2wJ3tkc-TU&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=14

1. Overfit the batch first

TO check wether the code is working or not

use `next.iter(train_dataloader)` to extract a single batch and thentrain for 1000 at batch size 1, to check if loss is actually going down.

2. while checking use model.eval() and model.train()

This removes stuff like batchnorma nd dropouts which is not required while doing inference

3. Dont forget to zero_grad() the optimizer

Because you want ot delete the accumaulated gardients of previous batch

4. dont use softmax with crossentropy loss

cross entropy loss in iteslef is two things its softmax followed by log liklihood

5. set bias =false with batchnorm

gives equivalent performance

6. dont confuse view with permute

they are not the same bro

7. Dont use wrong transformation for your dataset

dont transpose.RandomHorizontalFlip(p=1.0) with MNIST because that is going to change the digit entirely

8. always put shuffle=true while making your dataloader

except for in time based data like time series

9. NOrmalize the data

with transform.Nomalize(mean=(), std=())

10. in LSTM, GRU and RNN use torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1)

Because Alddin Perrson said so.

# LeNet implementation

https://www.kaggle.com/fanbyprinciple/pytorch-lenet-implementation/edit

Trying to implement the following architecture:

![](lenet_1.png)









