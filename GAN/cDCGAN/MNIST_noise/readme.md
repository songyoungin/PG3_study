# Conditional DCGAN implementation with Pytorch
## Dataset: MNIST

### Training hyper parameters
learning rate: 0.0002
num of training epochs: 20

### results
with decaying learning rate

![](https://github.com/younginsong21/PG3_study/blob/master/GAN/cDCGAN/MNIST_noise/samples/results.gif)

### comments
To solve the problem that generator loss doesn't decrease, I add a noise to input image of dicriminator.

But generated result is worse than that without adding noise.
