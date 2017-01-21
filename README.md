Deep learning model zoo
-----------------------

This repository contains implementations of various deep learning algorithms in Theano/Lasagne.

## Running a model

To run a model, you may use the `run.py` launch script.

```
python run.py train \
  --dataset <dataset> \
  --model <epochs> \
  --alg <opt_alg> \
  --n_batch <batch_size> \
  --lr <learning_rate> \
  -e <num_epochs> \
  -l <log_name>
```

Alternatively, you may use the `Makefile` included in the root dir; typing `make train` will start training. There are also several additional parameters that can be configured inside the `Makefile`.

The model will periodically save its weights and report training/validation losses in the logfile.

## Algorithms

The following algorithms are available.

### Supervised learning models

* `softmax`: simple softmax classifier
* `mlp`: multilayer perceptron
* `cnn`: convolutional neural network; solves `mnist` and achieves reasonably good accuracy on `cifar10`
* `resnet`: small residual network; achieves an accuracy in the 80's on `cifar10`

### Semi-supervised models

* `ssdadgm`: semi-supervised deep generative models (in progress)

### Unsupervised models

* `vae`: variational autoencoder
* `sbn`: sigmoid belief network trained with neural variational inference
* `adgm`: auxiliary deep generative model (unsupervised version)
* `dadgm`: discrete-variable auxiliary deep generative model (unsupervised version, also trained with NVIL)
* `dcgan`: small deep convolutional generative adversarial network (tested on mnist)

## Datasets

The following datasets are currently available:

* `cifar10`: color images divided into 10 classes (32x32x3)
* `mnist`: standard handwritten digits dataset (28x28)
* `digits`: sklearn digits dataset (8x8); can be used for quick debugging on a CPU

## Optimization methods

Currently, we may train the models using:

* `sgd`: standard stochastic gradient descent
* `adam`: the Adam optimizer

## Feedback

Send feedback to [Volodymyr Kuleshov](http://www.stanford.edu/~kuleshov). Some models contain snippets from other users' repositories; let me know if I forgot to cite anyone.
