# loading packages
from __future__ import print_function

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.models import Sequential
from matplotlib import pyplot as plt
from utilities.vis_utils import view_images
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

# training params
learning_rate = 1e-4
num_epochs = 15
batch_size = 128

# network parameters
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)
num_classes = 10
pool_size = (2, 2)
keep_prob = 1.0

# ensuring reproducibility
np.random.seed(42)

def load_mnist(view_grid=False):

	# loading the data: splitting into train and test
	(X_train, y_train), (X_test, y_test) = mnist.load_data()

	# grab dimensions for reshaping
	num_training = X_train.shape[0]
	num_test = X_test.shape[0]

	# # sanity check on the dimensions
	# print('Train Data Shape: {}'.format(X_train.shape))
	# print('Train Labels Shape: {}'.format(y_train.shape))
	# print('Test Data Shape: {}'.format(X_test.shape))
	# print('Test Labels Shape: {}'.format(y_test.shape))

	if view_grid:
		# let's view a grid of the images
		mask = range(0, 30)
		sample = X_train[mask]
		view_images(sample)

	# reshape data into rows
	X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
	X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

	# # recheck shape
	# print('X_train shape: {}'.format(X_train.shape))

	# normalize to [0-1]
	X_train /= 255
	X_test /= 255

	# convert target vectors to binary class matrices
	y_train = np_utils.to_categorical(y_train, num_classes)
	y_test = np_utils.to_categorical(y_test, num_classes)

	return X_train, y_train, X_test, y_test

# load the data
X_train, y_train, X_test, y_test = load_mnist()

# construct the model architecture
model = Sequential()

# ======================= 1st convolutional layer ========================

# apply a 5x5 convolution with 32 output filters on a 28x28 grayscale image
model.add(Convolution2D(32, 5, 5, init='he_normal', 
								  border_mode='same',
								  bias=True,
								  input_shape=input_shape))
# now output shape is (None, 32, 28, 28)

# batch-norm layer
model.add(BatchNormalization())
# output shape is same as input

# relu activation
model.add(Activation('relu'))
# output shape is same as input

# apply a pooling layer with filters of size 2x2 with a stride of 2
model.add(MaxPooling2D(pool_size=pool_size))
# now output shape is (None, 32, 14, 14)

# ======================= 2nd convolutional layer ========================

# apply a 5x5 convolution with 64 output filters on 14x14x32
model.add(Convolution2D(64, 5, 5, init='he_normal',
								  border_mode='same',
								  bias=True))

# now output shape is (None, 64, 14, 14)

# batch-norm layer
model.add(BatchNormalization())
# output shape is same as input

# relu activation
model.add(Activation('relu'))
# output shape is same as input

# apply same pooling layer as above
model.add(MaxPooling2D(pool_size=pool_size))
# now output shape is (None, 64, 7, 7)

# ======================= fully-connected layer ========================

# flatten the pooling output from (None, 64, 7, 7) to
model.add(Flatten())
# now output shape is (None, 64*7*7=3136)

# fully-connected layer
model.add(Dense(1024))
# now output shape is (None, 1024)

# batch-norm layer
model.add(BatchNormalization())
# output shape is same as input

# relu activation
model.add(Activation('relu'))
# output shape is same as input

# ======================= dropout layer ========================

model.add(Dropout(keep_prob))
# output shape is same as input

# ======================= output layer ========================

# fully-connected layer of size num_classes (i.e. 10)
model.add(Dense(num_classes))
# now output shape is (None, 10)

# softmax activation
model.add(Activation('softmax'))

# final summary of the model
model.summary()

# ====================================================== #

# define optimizer
adam = Adam(lr=learning_rate)

# compile the model
model.compile(loss='categorical_crossentropy', 
			  optimizer=adam, 
			  metrics=['accuracy'])

# train it
model.fit(X_train, y_train, 
				   batch_size=batch_size, 
				   nb_epoch=num_epochs,
				   verbose=1, 
				   validation_data=(X_test, y_test))

# evaluate on test set
score = model.evaluate(X_test, y_test, verbose=0)

# print accuracy
print('Test score:', score[0])
print('Test accuracy:', score[1])

model.save('mnist_convnet.h5')
