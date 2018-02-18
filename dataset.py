import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np


class DataSet(object):
  def __init__(self, images, labels, img_names):
    self._images = images
    self._labels = labels
    self._img_names = img_names

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def img_names(self):
    return self._img_names


class DataSets(object):
  def __init__(self, training, validation):
      self._training = training
      self._validation = validation

  @property
  def training(self):
      return self._training

  @property
  def validation(self):
      return self._validation


def read_train_sets(training_path, image_size, classes, validation_size):
    training_images = []
    training_labels = []
    training_img_names = []

    validation_images = []
    validation_labels = []
    validation_img_names = []

    print('Going to read training images\n')
    for fields in classes:
        index = classes.index(fields)
        print('\tGoing to read {} files (Index: {})'.format(fields, index))

        path = os.path.join(training_path, fields, '*.png')
        print('\tPath: ' + path)
        files = glob.glob(path)
        print('\tNumber of files: ' + str(files.__len__()) + '\n')

        images_for_class = []
        labels_for_class = []
        img_names_for_class = []

        for fl in files:
            image = cv2.imread(fl, 0)
            image = cv2.resize(image, (image_size, image_size))
            imgarr = np.array(image).astype(float).flatten()
            images_for_class.append(imgarr / 255)

            label = np.zeros(len(classes))
            label[index] = 1.0
            labels_for_class.append(label)

            flbase = os.path.basename(fl)
            img_names_for_class.append(flbase)

        validation_len = int(validation_size * images_for_class.__len__())

        # First n = validation_len images from each class are in validation set
        validation_images.extend(images_for_class[:validation_len])
        validation_labels.extend(labels_for_class[:validation_len])
        validation_img_names.extend(img_names_for_class[:validation_len])

        # All images except first n = validation_len from each class are in training set
        training_images.extend(images_for_class[validation_len:])
        training_labels.extend(labels_for_class[validation_len:])
        training_img_names.extend(img_names_for_class[validation_len:])

    training_images, training_labels, training_img_names = shuffle(training_images, training_labels, training_img_names)
    t_images = np.array(training_images)
    t_labels = np.array(training_labels)
    t_img_names = np.array(training_img_names)

    validation_images, validation_labels, validation_img_names = shuffle(validation_images, validation_labels, validation_img_names)
    v_images = np.array(validation_images)
    v_labels = np.array(validation_labels)
    v_img_names = np.array(validation_img_names)

    train = DataSet(t_images, t_labels, t_img_names)
    valid = DataSet(v_images, v_labels, v_img_names)
    data_sets = DataSets(train, valid)

    return data_sets
