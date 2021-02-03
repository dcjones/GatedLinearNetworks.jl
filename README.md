
An implementation of [Gaussian Gated Linear Networks](https://arxiv.org/abs/2006.05964), an interesting type of neural network that doesn't use backpropagation.

It can fit arbitrary continuous functions, a simple example of which is below:
![example fit to noisy spiral function](https://github.com/dcjones/GatedLinearNetworks.jl/blob/main/example.png?raw=true)

Things left to do:
  * Optimize (it's super slow right now)
  * Add GPU support (easy in theory)

