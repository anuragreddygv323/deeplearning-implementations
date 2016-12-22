# loading packages
from __future__ import print_function

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.models import Sequential
from utilities.vis_utils import view_images
from keras.layers import Dense, Activation, Dropout

# training params
learning_rate = 1e-3
num_epochs = 20
batch_size = 128

# network parameters
num_nodes_l1 = 256
num_nodes_l2 = 256
num_nodes_l3 = 256
num_input = 784
num_classes = 10

# ensuring reproducibility
np.random.seed(42)

def load_mnist(view_grid=False):
	"""
	Helper function for preprocessing the MNIST dataset.

	Performs the following:
	- reshapes the data into rows.
	- cast as float32.
	- normalize to [0-1].
	- 1-hot encodes the target vector.

	Also provides optional viewing of the dataset.
	"""

	# loading the data: splitting into train and test
	(X_train, y_train), (X_test, y_test) = mnist.load_data()

	# grab dimensions for reshaping
	num_training = X_train.shape[0]
	num_test = X_test.shape[0]

	# sanity check on the dimensions
	print('Train Data Shape: {}'.format(X_train.shape))
	print('Train Labels Shape: {}'.format(y_train.shape))
	print('Test Data Shape: {}'.format(X_test.shape))
	print('Test Labels Shape: {}'.format(y_test.shape))

	if view_grid:
		# let's view a grid of the images
		mask = range(0, 30)
		sample = X_train[mask]
		view_images(sample)

	# reshape data into rows and cast as float32.
	X_train = X_train.reshape(num_training, -1).astype('float32')
	X_test = X_test.reshape(num_test, -1).astype('float32')

	# normalize to [0-1]
	X_train /= 255
	X_test /= 255

	# 1-hot encoding
	y_train = np_utils.to_categorical(y_train, num_classes)
	y_test = np_utils.to_categorical(y_test, num_classes)

	return X_train, y_train, X_test, y_test

# load the data
X_train, y_train, X_test, y_test = load_mnist(view_grid=True)

# declaring three layer fully-connected net architecture
model = Sequential()

# input layer
model.add(Dense(num_nodes_l1, input_dim=num_input))
model.add(Activation('relu'))

# 1st hidden layer
model.add(Dense(num_nodes_l2))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# 2nd hidden layer
model.add(Dense(num_nodes_l3))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# 3rd hidden layer (i.e. output)
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# summary of the architecture
model.summary()

# define optimizer and compile model
sgd = SGD(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# train the model
history = model.fit(X_train, 
                  y_train,
                  batch_size=batch_size, 
                  nb_epoch=num_epochs,
                  verbose=1, 
                  validation_data=(X_test, y_test))

# evaluate the model
score = model.evaluate(X_test, y_test)
print('\nTest score: {}'.format(score[0]))
print('Test accuracy: {}'.format(score[1]))
