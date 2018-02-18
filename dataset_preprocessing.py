import cv2
import os
import glob
import numpy as np


def image_preprocessing(image):
    # create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    img = clahe.apply(image)
    img = np.float32(img)

    # Gabor filter angles (degrees): 0, 22.5 ,45, 67.5, 90 ,112.5, 135 ,157.5
    # Gabor filter angles (radians): 0, 0.3926990817 0.7853981634 1.178 1.570 1.9634954085 2.3561944902 2.7488935719

    # cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
    # ksize - size of gabor filter (n, n)
    # sigma - standard deviation of the gaussian function
    # theta - orientation of the normal to the parallel stripes
    # lambda - wavelength of the sunusoidal factor
    # gamma - spatial aspect ratio
    # psi - phase offset
    # ktype - type and range of values that each pixel in the gabor kernel can hold

    g_kernel_1 = cv2.getGaborKernel((21, 21), 8.0, 1.570, 10.0, 0.5, 0, ktype=cv2.CV_32F)  # 0
    g_kernel_2 = cv2.getGaborKernel((21, 21), 8.0, 1.178, 10.0, 0.5, 0, ktype=cv2.CV_32F)  # 22.5
    g_kernel_3 = cv2.getGaborKernel((21, 21), 8.0, 0.7853981634, 10.0, 0.5, 0, ktype=cv2.CV_32F)  # 45
    g_kernel_4 = cv2.getGaborKernel((21, 21), 8.0, 0.3926990817, 10.0, 0.5, 0, ktype=cv2.CV_32F)  # 67.5
    g_kernel_5 = cv2.getGaborKernel((21, 21), 8.0, 0, 10.0, 0.5, 0, ktype=cv2.CV_32F)  # 90
    g_kernel_6 = cv2.getGaborKernel((21, 21), 8.0, 2.7488935719, 10.0, 0.5, 0, ktype=cv2.CV_32F)  # 112.5
    g_kernel_7 = cv2.getGaborKernel((21, 21), 8.0, 2.3561944902, 10.0, 0.5, 0, ktype=cv2.CV_32F)  # 135
    g_kernel_8 = cv2.getGaborKernel((21, 21), 8.0, 1.9634954085, 10.0, 0.5, 0, ktype=cv2.CV_32F)  # 157.5

    gabor1 = cv2.filter2D(img, cv2.CV_32F, g_kernel_1)
    gabor2 = cv2.filter2D(img, cv2.CV_32F, g_kernel_2)
    gabor3 = cv2.filter2D(img, cv2.CV_32F, g_kernel_3)
    gabor4 = cv2.filter2D(img, cv2.CV_32F, g_kernel_4)
    gabor5 = cv2.filter2D(img, cv2.CV_32F, g_kernel_5)
    gabor6 = cv2.filter2D(img, cv2.CV_32F, g_kernel_6)
    gabor7 = cv2.filter2D(img, cv2.CV_32F, g_kernel_7)
    gabor8 = cv2.filter2D(img, cv2.CV_32F, g_kernel_8)

    enhanced = gabor1 + gabor2 + gabor3 + gabor4 + gabor5 + gabor6 + gabor7 + gabor8

    # Do all filters on one image - result image of 8 filters
    enhanced = cv2.addWeighted(enhanced, 0, gabor1, 1, 0)
    enhanced = cv2.addWeighted(enhanced, 1, gabor2, 1, 0)
    enhanced = cv2.addWeighted(enhanced, 1, gabor3, 1, 0)
    enhanced = cv2.addWeighted(enhanced, 1, gabor4, 1, 0)
    enhanced = cv2.addWeighted(enhanced, 1, gabor5, 1, 0)
    enhanced = cv2.addWeighted(enhanced, 1, gabor6, 1, 0)
    enhanced = cv2.addWeighted(enhanced, 1, gabor7, 1, 0)
    enhanced = cv2.addWeighted(enhanced, 1, gabor8, 1, 0)

    # # Printing images with kernel used in Gabor filters
    # # 1
    # h1, w1 = g_kernel_1.shape[:2]
    # g_kernel_1 = cv2.resize(g_kernel_1, (3 * w1, 3 * h1), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('gabor kernel1 (resized)', g_kernel_1)
    # # 2
    # h2, w2 = g_kernel_2.shape[:2]
    # g_kernel_2 = cv2.resize(g_kernel_2, (3 * w2, 3 * h2), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('gabor kernel_2 (resized)', g_kernel_2)
    # # 3
    # h3, w3 = g_kernel_3.shape[:2]
    # g_kernel_3 = cv2.resize(g_kernel_3, (3 * w3, 3 * h3), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('gabor kernel3 (resized)', g_kernel_3)
    # # 4
    # h4, w4 = g_kernel_4.shape[:2]
    # g_kernel_4 = cv2.resize(g_kernel_4, (3 * w4, 3 * h4), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('gabor kernel4 (resized)', g_kernel_4)
    # # 5
    # h5, w5 = g_kernel_5.shape[:2]
    # g_kernel_5 = cv2.resize(g_kernel_5, (3 * w5, 3 * h5), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('gabor kernel5 (resized)', g_kernel_5)
    # # 6
    # h6, w6 = g_kernel_6.shape[:2]
    # g_kernel_6 = cv2.resize(g_kernel_6, (3 * w6, 3 * h6), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('gabor kernel6 (resized)', g_kernel_6)
    # # 7
    # h7, w7 = g_kernel_7.shape[:2]
    # g_kernel_7 = cv2.resize(g_kernel_7, (3 * w7, 3 * h7), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('gabor kernel7 (resized)', g_kernel_7)
    # # 8
    # h8, w8 = g_kernel_8.shape[:2]
    # g_kernel_8 = cv2.resize(g_kernel_8, (3 * w8, 3 * h8), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('gabor kernel8 (resized)', g_kernel_8)

    return enhanced


def preprocess_images_from_dataset(dataset_path, classes, new_path):
    print('Going to read dataset images\n')
    for fields in classes:
        index = classes.index(fields)
        print('\tGoing to read {} files (Index: {})'.format(fields, index))

        path = os.path.join(dataset_path, fields, '*.png')
        files = glob.glob(path)
        print('\tNumber of files: ' + str(files.__len__()) + '\n')

        for fl in files:
            image = cv2.imread(fl, 0)
            image = image_preprocessing(image)
            image_path_tokens = fl.split('/')
            image_name = image_path_tokens[len(image_path_tokens)-1]
            cv2.imwrite(new_path + '/' + fields + '/' + image_name, image)