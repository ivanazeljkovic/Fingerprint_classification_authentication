import dataset
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import MaxPooling2D, Conv2D
from keras.optimizers import SGD


# Classes of fingerprints in dataset
classes = ['arch', 'left_loop', 'right_loop', 'tented_arch', 'whorl']
num_classes = len(classes)


# 20% of the data will automatically be used for validation
validation_size = 0.2
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


# Building 'VGG-19 Network'
model = Sequential()
input_shape = (img_size, img_size, num_channels)

model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# convolution layer 3 and 4 in block 3 are ignored, because of network size and training
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# convolution layer 3 and 4 in block 4 are ignored, because of network size and training
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# convolution layer 3 and 4 in block 5 are ignored, because of network size and training
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(training_X, training_Y, batch_size=64, nb_epoch=150, verbose=1,
          validation_data=(validation_X, validation_Y))

model.save("model/vgg-19-model-150-epoch")
