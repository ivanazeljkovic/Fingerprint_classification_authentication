import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np
import dataset_preprocessing
import keras


# Classes of fingerprints in dataset
classes = ['arch', 'left_loop', 'right_loop', 'tented_arch', 'whorl']
# 224 for ZFNet and 227 for AlexNet
img_size = 224


def find_max_probability(array):
    return array.tolist().index(max(array))


def load_all_testing_paths():
    paths = []
    for fields in classes:
        path = os.path.join('data/testing', fields, '*.png')
        files = glob.glob(path)
        paths.extend(files)

    paths = shuffle(paths)
    return paths


def load_class(path):
    # Path to appropriate txt file with class/classes for selected image
    path = path[:-3] + 'txt'

    file = open(path, 'r')
    lines = file.readlines()
    line = lines[len(lines) - 1]
    classes = line.split(' ')[2]

    posible_classes = []
    posible_classes.append(classes[0])
    if(len(classes) > 1):
        posible_classes.append(classes[1])

    return posible_classes


def load_model():
    # model/zfnet-model-300-epoch for ZFNet, model/alex-net-model-300-epoch for AlexNet
    model = keras.models.load_model("model/zfnet-model-300-epoch")
    return model


def predict():
    model = load_model()
    test_paths = load_all_testing_paths()
    number_of_test_el = len(test_paths)
    number_of_good_predictions = 0

    for path in test_paths:
        expected_classes = load_class(path)
        print(path)

        image = cv2.imread(path, 0)
        image = dataset_preprocessing.image_preprocessing(image)
        image = cv2.resize(image, (img_size, img_size))
        imgarr = np.array(image).astype(float)
        for i in range(len(imgarr)):
            imgarr[i] = imgarr[i] / 255.0

        image_for_prediction = imgarr.reshape(1, img_size, img_size, 1)
        result = model.predict(image_for_prediction)
        index_of_class = find_max_probability(result[0])

        predicted_class = classes[index_of_class]
        if str.upper(predicted_class[0]) in expected_classes:
            number_of_good_predictions += 1

        print('Predicted class: ' + str.upper(predicted_class[0]) + ' --- Expected class/classes: ' + expected_classes[0] +
              (expected_classes[1] if len(expected_classes) > 1 else ''))

    percentage = (number_of_good_predictions / float(number_of_test_el)) * 100
    print('Percentual success: %.2f' % percentage)


def predict_class_for_image(image_path):
    model = load_model()

    expected_classes = load_class(image_path)
    print(image_path)

    image = cv2.imread(image_path, 0)
    image = dataset_preprocessing.image_preprocessing(image)
    image = cv2.resize(image, (img_size, img_size))
    imgarr = np.array(image).astype(float)
    for i in range(len(imgarr)):
        imgarr[i] = imgarr[i] / 255.0

    image_for_prediction = imgarr.reshape(1, img_size, img_size, 1)
    result = model.predict(image_for_prediction)
    index_of_class = find_max_probability(result[0])

    predicted_class = classes[index_of_class]
    if str.upper(predicted_class[0]) in expected_classes:
        return True
    return False