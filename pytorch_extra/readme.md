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
2. cannot display proper images

![](learning_pytorch_8_error.png)

check code here :
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/custom_dataset_txt/loader_customtext.py


https://www.kaggle.com/fanbyprinciple/learning-pytorch-8-working-with-text-dataset/edit

## Augmenting dataset using torchvision

https://www.youtube.com/watch?v=Zvd276j9sZ8&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=11


Image dataset loaded

![](learning_pytorch_9_image_dataset_loaded.png)

https://www.kaggle.com/fanbyprinciple/learning-pytorch-9-data-augmentation-torchvision/edit





