import dataset
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import MaxPooling2D, Conv2D, BatchNormalization
from keras.optimizers import SGD


# Classes of fingerprints in dataset
classes = ['arch', 'left_loop', 'right_loop', 'tented_arch', 'whorl']
num_classes = len(classes)


# 20% of the data will automatically be used for validation
validation_size = 0.2
# 224 for ZFNet and 227 for AlexNet
img_size = 224
num_channels = 1
training_path = 'processed_data/training'


# Load all the training (and validation) images and labels into memory using openCV and use that during training
data = dataset.read_train_sets(training_path, img_size, classes, validation_size)

print("\nComplete reading input data\n")
print("\tNumber of files in Training-set:\t{}".format(len(data.training.labels)))
print("\tNumber of files in Validation-set:\t{}".format(len(data.validation.labels)))


training_X = data.training.images.reshape([-1, img_size, img_size, 1])
training_Y = data.training.labels
validation_X = data.validation.images.reshape([-1, img_size, img_size, 1])
validation_Y = data.validation.labels


# Building 'AlexNet' / 'ZFNet'
model = Sequential()

# First convolution layer for AlexNet
# model.add(Conv2D(96, (11, 11), strides=(4, 4), activation='relu', padding='same', input_shape=(img_size, img_size, num_channels)))
# First convolution layer for ZFNet
model.add(Conv2D(96, (7, 7), strides=(2, 2), activation='relu', padding='same', input_shape=(img_size, img_size, num_channels)))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
# Local Response normalization for original AlexNet
model.add(BatchNormalization())

model.add(Conv2D(256, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
# Local Response normalization for original AlexNet
model.add(BatchNormalization())

model.add(Conv2D(384, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(384, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
# Local Response normalization for original AlexNet
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(784, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(784, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

sgd = SGD(lr=0.001, decay=1.e-6, momentum=0.9, nesterov=False)

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(training_X, training_Y, batch_size=64, nb_epoch=300, verbose=1,
          validation_data=(validation_X, validation_Y))

model.save("model/zfnet-model-300-epoch")
