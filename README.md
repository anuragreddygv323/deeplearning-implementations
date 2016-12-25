## Deep Learning Implementations Roadmap in Keras

#### Basics

- [x] [Fully-Connected Neural Network](https://github.com/kevinzakka/keras_practice/blob/master/fully_connected_net.py)
	- **Dataset**: MNIST
	- **Architecture**: 3 hidden layers, ReLU activations and Dropout.
	
- [x] [Convolutional Neural Network](https://github.com/kevinzakka/keras_practice/blob/master/conv_net_mnist.py)
	- **Dataset**: MNIST
	- **Architecture**: (Conv - Batch - ReLU - MaxPool)*2 - (FC - Batch - ReLU - Dropout) - (FC - Softmax)
	- **Conv Layers**: 32 (5x5) filters with stride of 1 [layer 1], 64 (5x5) filters with stride of 1 [layer 2]. Zero-padded for both.
	- **Pooling Layers**: 2x2 filters with a stride of 2.
	- **Learning**: adam optimizer, 1e-4 learning rate, batch size of 128, 15 epochs.

#### Image Processing

- [X] Neural Styler
	- [arXiv](https://arxiv.org/abs/1508.06576)
	- Code coming soon - (pruning and fixing)
- [ ] Fast Neural Style
- [ ] Generative Adversarial Network
- [ ] Spatial Transformer Networks
- [ ] Pixel Recurrent Neural Networks
- [ ] Conditional PixelCNN Decoder
- [ ] Video Pixel Networks
- [ ] DRAW

#### Image Segmentation

- [ ] U-Net
- [ ] Deepmask, SharpMask and MultiPathNet

#### Natural Language Processing

- [ ] RNN
- [ ] LSTM
- [ ] GRU
- [ ] ByteNet

#### Sound

- [ ] Wavenet

