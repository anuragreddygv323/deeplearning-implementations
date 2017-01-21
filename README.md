## Deep Learning Paper Implementations

- [X] [Fully-Connected Neural Network](https://github.com/kevinzakka/keras_practice/blob/master/fully_connected_net.py)
	- **Dataset**: MNIST
	- **Architecture**: 3 hidden layers, ReLU activations and Dropout.
	- **Framework**: Keras
	
- [X] [Convolutional Neural Network](https://github.com/kevinzakka/keras_practice/blob/master/conv_net_mnist.py)
	- **Dataset**: MNIST
	- **Architecture**: (Conv - Batch - ReLU - MaxPool)*2 - (FC - Batch - ReLU - Dropout) - (FC - Softmax)
	- **Conv Layers**: 32 (5x5) filters with stride of 1 [layer 1], 64 (5x5) filters with stride of 1 [layer 2]. Zero-padded for both.
	- **Pooling Layers**: 2x2 filters with a stride of 2.
	- **Learning**: adam optimizer, 1e-4 learning rate, batch size of 128, 15 epochs.
	- **Framework**: Keras
	
- [X] [Neural Styler](https://github.com/kevinzakka/style_transfer)
	- **Framework**: Keras
- [X] [Spatial Transformer Networks](https://github.com/kevinzakka/spatial_transformer_network)
	- Summary
	- **Framework**: Tensorflow

## Up Next

- Fast Neural Styler
	- Summary
- Bytenet
	- [Summary](https://github.com/kevinzakka/research-paper-notes/blob/master/linear_time_nmt.md)
- Molecular Autoencoder
	- Summary
