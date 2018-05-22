# Project : Deep Learning Quadrotor Follower

In this project, I built a sementic segmentation network with an FCN; the aim was to control a quadrotor in a simulated urban environment to follow a target among a dense crowd.

See the [Youtube Video](https://youtu.be/HBVXjhB7Al4) for the implementation in action.

## Network Architecture

The network followed a standard setup of an hourglass network composed of encoding and decoding stages,
with skip connections in between to utilize high-information features at earlier layers. In doing so, it mostly followed the provided template.

Each encoder block was composed of a single 1x1 convolution with half of the output filter size for efficiency,
followed by another separable convolution with 3x3 kernels with the respective output filter size.

Each decoder block was composed of an upsampled previous layer concatenated with the larger input from earlier encoding stages,
followed by two-step separable convolutions similar to the encoder block.

## Network Tuning

![figs/train.png](figs/train.png)

After tuning the network, the final score was 0.474.

### Hyperparameters

In the final training session of the model, I used the following set of hyperparameters:

| Parameter        | Value |
|:----------------:|:-----:|
| Learning Rate    | 1e-3  |
| LR Decay         | 2e-4  |
| Batch Size       | 32    |
| Epochs           | 50    |
| Epoch Steps      | 100   |
| Validation Steps | 50    |
| Workers          | 4     |

#### Learning Rate

The ballpark learning rate was set to be about 1e-3 from experience. However, instead of a fixed learning rate, the learning rate was decayed over time in order to promote stability and convergence.

Keras supports the following decay scheme by default:

```python
lr = lr * (1. / (1. + decay * n))
```

Where n is the total update steps from the beginning.
Accordingly, the decay was determined as;

```python
total_steps = steps_per_epoch * num_epochs
lr_decay = (1.0 / 0.5 - 1.0) / total_steps
```

Or 2e-4, in order to anneal the learning rate from 1e-3 to half (5e-4) by the end.

#### Batch Size

The batch size was determined partly based on experience; batch size of 32 proved sufficiently effective in mitigating the effects of determining the gradient from a small number of observed sample from a step, while maintaining the speed benefits of not having to iterate over the entire training set.

#### Epochs

In terms of the model training, what is really critical is the total number of steps, i.e. the number of epochs multiplied by the number of steps in each epoch.

As far as monitoring progress goes, I decided that it is sufficient to look at the results every 100 steps, and based on the runs it seemed that the network converged after approximately 5000 steps. The number of epochs were determined accordingly.


### Regularization

Based on the initial runs, there was strong evidence of the network overfitting to the training samples.

In order to mitigate this issue, I introduced 0.2 dropout in the final encoding after the 1x1 convolution,

as well as l2 regularization with 1e-4 weights to all of the trainable parameters in each convolution stages.

This way, the network learned to better generalize to the samples, rather than rote memorizization of the training samples.