# Deep dive into Convolutions

## Importance

1. It tells us exactly how convolutional operation takes place over a filter.

1. tells about how to design a simple cnn network.

1. tells about how to use hooks to look into training data

1. how 1 cycle training works

## CNN

CNN filters can be made for edge detections:
![filter image](./img/convolution_filters.png)

Convolution notebook also features cute bears for color convolutions:
![multicolored bear](./img/bear_colors.png)

Simple baseline modelnotebook tells us about how to use hooks and then tells us why it is important to have better batch size, simple one cycle training, batch normalisation

The problem with normal cnn that we make is that due to large number of zeros barely any zeros reach the penultimate layer
![last layer](penultimate_activations.png)

### 1 cycle training

1cycle training allows us to use a much higher maximum learning rate than other types of training, which gives two benefits:

By training with higher learning rates, we train faster—a phenomenon Smith named super-convergence.
By training with higher learning rates, we overfit less because we skip over the sharp local minima to end up in a smoother (and therefore more generalizable) part of the loss.

Read about it.