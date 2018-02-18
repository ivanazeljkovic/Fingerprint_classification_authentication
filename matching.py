import cv2
import numpy as np
from skimage.morphology import skeletonize
import math


def make_thin(image_path):
    img = cv2.imread(image_path, 0)
    img = cv2.bitwise_not(img)
    ret, img = cv2.threshold(img, 127, 255, 0)

    ske = (skeletonize(img // 255) * 255).astype(np.uint8)

    return ske


def extract_minutiae_from_image(image):
    height, width = image.shape
    image = (image / 255).astype(np.uint8)

    # coordinates of all black pixels
    black_pixels = []

    # ignore all black pixels in border 10 pixels => prevent extracting false minutiae
    # image rows
    for i in range(10, height - 10):
        # image columns
        for j in range(10, width - 10):
            if image[i, j] == 0:
                black_pixels.append([i, j])

    # coordinates of black pixels that represent bifurcation
    bifurcation_list = []

    # coordinates of black pixels that represent ridge ending
    ridge_ending_list = []

    # Crossing number algorithm for matrix of 8 neighbour
    # P4 | P3 | P2
    # ------------
    # P5 | P  | P1
    # ------------
    # P6 | P7 | P8

    for black_pixel in black_pixels:
        row = black_pixel[0]
        column = black_pixel[1]

        P1 = int(image[row, column + 1])
        P2 = int(image[row - 1, column + 1])
        P3 = int(image[row - 1, column])
        P4 = int(image[row - 1, column - 1])
        P5 = int(image[row, column - 1])
        P6 = int(image[row + 1, column - 1])
        P7 = int(image[row + 1, column])
        P8 = int(image[row + 1, column + 1])
        P9 = P1

        CN = 1/2 * (abs(P1 - P2) + abs(P2 - P3) + abs(P3 - P4) + abs(P4 - P5) + abs(P5 - P6) + abs(P6 - P7) + abs(P7 - P8) + abs(P8 - P9))

        # crossing number value is 3 for bifurcation and 1 for ridge ending
        if CN == 3:
            bifurcation_list.append([row, column])
        elif CN == 1:
            ridge_ending_list.append([row, column])

    print('\nBefore remove -> bifurcation list: ' + str(len(bifurcation_list)))
    print('Before remove -> ridge ending list: ' + str(len(ridge_ending_list)))
    print('Before -> all: ' + str(len(bifurcation_list) + len(ridge_ending_list)) + '\n')

    img = draw_minutiae(image, bifurcation_list, ridge_ending_list, black_pixels)
    remove_false_minutiae(bifurcation_list, ridge_ending_list)
    print('After remove -> bifurcation list: ' + str(len(bifurcation_list)))
    print('After remove -> ridge ending list: ' + str(len(ridge_ending_list)))
    print('After -> all: ' + str(len(bifurcation_list) + len(ridge_ending_list)) + '\n')
    img1 = draw_minutiae(image, bifurcation_list, ridge_ending_list, black_pixels)
    return img, img1, bifurcation_list, ridge_ending_list


def check_matching(image_path, image_path2):
    image = make_thin(image_path)
    image2 = make_thin(image_path2)

    image = cv2.bitwise_not(image)
    image2 = cv2.bitwise_not(image2)
    # cv2.imshow('image', image)
    # cv2.imshow('image2', image2)

    img1, img2, bifurcation_list1, ridge_ending_list1 = extract_minutiae_from_image(image)
    img3, img4, bifurcation_list2, ridge_ending_list2 = extract_minutiae_from_image(image2)

    match_minutiae(img1, bifurcation_list1, ridge_ending_list1, img3, bifurcation_list2, ridge_ending_list2)

    # cv2.imshow('image1 with all minutiae', img1)
    # cv2.imshow('image1 with right minutiae', img2)
    # cv2.imshow('image2 with all minutiae', img3)
    # cv2.imshow('image2 with right minutiae', img4)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def draw_minutiae(image, bifurcation_list, ridge_ending_list, black_pixels):
    img = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
    for black_pixel in black_pixels:
        img[black_pixel[0]][black_pixel[1]] = (255, 255, 255)

    img = cv2.bitwise_not(img)

    for i in bifurcation_list:
        cv2.circle(img, (i[1], i[0]), 2, (0, 0, 255), -1)

    for j in ridge_ending_list:
        cv2.circle(img, (j[1], j[0]), 2, (255, 0, 0), -1)

    return img


def euclidean_distance(point1, point2):
    return math.sqrt(math.pow((point1[0]-point2[0]), 2) + math.pow((point1[1]-point2[1]), 2))


def remove_false_minutiae(bifurcation_list, ridge_ending_list):
    # value empirically set to 6
    distance_threshold = 6.0

    false_bifurcations_index = []
    false_bifurcations = []

    # Check bifurcation and bifurcation pairs
    for bifurcation in bifurcation_list:
        index = bifurcation_list.index(bifurcation)
        # only if we didn't already pick bifurcation as false
        if index not in false_bifurcations_index:
            # use matrix 11x11 which center is selected bifurcation
            min_i = bifurcation[0] - 5
            max_i = bifurcation[0] + 5
            min_j = bifurcation[1] - 5
            max_j = bifurcation[1] + 5

            for bifurcation_neighbour in bifurcation_list:
                i = bifurcation_neighbour[0]
                j = bifurcation_neighbour[1]
                index_neighbour = bifurcation_list.index(bifurcation_neighbour)

                # only if we didn't already pick bifurcation as false
                if index_neighbour not in false_bifurcations_index:
                    # if selected bifurcation is not current bifurcation which we check
                    # and if selected bifurcation has coordinates in range [min_i, max_i], [min_j, max_j]
                    if (index_neighbour != index) & ((i >= min_i) & (i <= max_i)) & ((j >= min_j) & (j <= max_j)):
                        distance = euclidean_distance(bifurcation, bifurcation_neighbour)
                        if distance < distance_threshold:
                            if index not in false_bifurcations_index:
                                false_bifurcations_index.append(index)
                                false_bifurcations.append([bifurcation[0], bifurcation[1]])
                            if index_neighbour not in false_bifurcations_index:
                                false_bifurcations_index.append(index_neighbour)
                                false_bifurcations.append([bifurcation_neighbour[0], bifurcation_neighbour[1]])

    for bifurcation in false_bifurcations:
        index = -1
        for bif in bifurcation_list:
            if (bifurcation[0] == bif[0]) & (bifurcation[1] == bif[1]):
                index = bifurcation_list.index(bif)
                break
        if index != -1:
            del bifurcation_list[index]

    print('False bifurcations: ' + str(len(false_bifurcations)))



    false_ridge_endings_index = []
    false_ridge_endings = []
    # Check ridge ending and ridge ending pairs
    for ridge_ending in ridge_ending_list:
        index = ridge_ending_list.index(ridge_ending)
        # only if we didn't already pick ridge ending as false
        if index not in false_ridge_endings_index:
            # use matrix 11x11 which center is selected ridge ending
            min_i = ridge_ending[0] - 5
            max_i = ridge_ending[0] + 5
            min_j = ridge_ending[1] - 5
            max_j = ridge_ending[1] + 5

            for ridge_ending_neighbour in ridge_ending_list:
                i = ridge_ending_neighbour[0]
                j = ridge_ending_neighbour[1]
                index_neighbour = ridge_ending_list.index(ridge_ending_neighbour)

                # only if we didn't already pick ridge ending as false
                if index_neighbour not in false_ridge_endings_index:
                    # if selected ridge ending is not current ridge ending which we check
                    # and if selected ridge ending has coordinates in range [min_i, max_i], [min_j, max_j]
                    if (index_neighbour != index) & ((i >= min_i) & (i <= max_i)) & ((j >= min_j) & (j <= max_j)):
                        distance = euclidean_distance(ridge_ending, ridge_ending_neighbour)
                        if distance < distance_threshold:
                            if index not in false_ridge_endings_index:
                                false_ridge_endings_index.append(index)
                                false_ridge_endings.append([ridge_ending[0], ridge_ending[1]])
                            if index_neighbour not in false_ridge_endings_index:
                                false_ridge_endings_index.append(index_neighbour)
                                false_ridge_endings.append([ridge_ending_neighbour[0], ridge_ending_neighbour[1]])

    for ridge_ending in false_ridge_endings:
        index = -1
        for rig in ridge_ending_list:
            if (ridge_ending[0] == rig[0]) & (ridge_ending[1] == rig[1]):
                index = ridge_ending_list.index(rig)
                break
        if index != -1:
            del ridge_ending_list[index]

    print('False ridge endings: ' + str(len(false_ridge_endings)))



    false_mixed_index = []
    false_mixed = []
    # Check ridge ending and bifurcation pairs
    for ridge_ending in ridge_ending_list:
        index = ridge_ending_list.index(ridge_ending)
        # only if we didn't already pick ridge ending as false
        if index not in false_mixed_index:
            # use matrix 11x11 which center is selected ridge ending
            min_i = ridge_ending[0] - 5
            max_i = ridge_ending[0] + 5
            min_j = ridge_ending[1] - 5
            max_j = ridge_ending[1] + 5

            for bifurcation_neighbour in bifurcation_list:
                i = bifurcation_neighbour[0]
                j = bifurcation_neighbour[1]
                index_neighbour = bifurcation_list.index(bifurcation_neighbour)

                # only if we didn't already pick bifurcation as false
                if index_neighbour not in false_mixed_index:
                    # if selected bifurcation is not current ridge ending which we check
                    # and if selected ridge ending has coordinates in range [min_i, max_i], [min_j, max_j]
                    if (index_neighbour != index) & ((i >= min_i) & (i <= max_i)) & ((j >= min_j) & (j <= max_j)):
                        distance = euclidean_distance(ridge_ending, bifurcation_neighbour)
                        if distance < distance_threshold:
                            if index not in false_mixed_index:
                                false_mixed_index.append(index)
                                false_mixed.append([ridge_ending[0], ridge_ending[1]])
                            if index_neighbour not in false_mixed_index:
                                false_mixed_index.append(index_neighbour)
                                false_mixed.append([bifurcation_neighbour[0], bifurcation_neighbour[1]])

    false_bif = 0
    false_rig = 0
    for false in false_mixed:
        index = -1
        is_ridge_ending = False

        for rig in ridge_ending_list:
            if (false[0] == rig[0]) & (false[1] == rig[1]):
                index = ridge_ending_list.index(rig)
                is_ridge_ending = True
                false_rig += 1
                break
        if index == -1:
            for bif in bifurcation_list:
                if (false[0] == bif[0]) & (false[1] == bif[1]):
                    index = bifurcation_list.index(bif)
                    is_ridge_ending = False
                    false_bif += 1
                    break
        if index != -1:
            if is_ridge_ending:
                del ridge_ending_list[index]
            else:
                del bifurcation_list[index]

    print('False bifurcation from mixed: ' + str(false_bif))
    print('False ridge ending from mixed: ' + str(false_rig) + '\n')


def match_minutiae(image1, bifurcation_list1, ridge_ending_list1, image2, bifurcation_list2, ridge_ending_list2):
    keypoints1 = []
    keypoints2 = []

    # store keypoints for first image
    for bifurcation in bifurcation_list1:
        keypoints1.append(cv2.KeyPoint(bifurcation[0], bifurcation[1], 1))

    for ridge_ending in ridge_ending_list1:
        keypoints1.append(cv2.KeyPoint(ridge_ending[0], ridge_ending[1], 1))

    # store keypoints for second image
    for bifurcation in bifurcation_list2:
        keypoints2.append(cv2.KeyPoint(bifurcation[0], bifurcation[1], 1))

    for ridge_ending in ridge_ending_list2:
        keypoints2.append(cv2.KeyPoint(ridge_ending[0], ridge_ending[1], 1))

    # Initiate ORB detector
    orb = cv2.ORB_create()

    # find the descriptors for given keypoints
    kp1, des1 = orb.compute(image1, keypoints1)
    kp2, des2 = orb.compute(image2, keypoints2)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw all matches
    # img3 = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=2)
    #
    # plt.imshow(img3), plt.show()

    print('Number of matches: ' + str(len(matches)))
    success_of_matches1 = (len(matches) / len(keypoints1)) * 100
    success_of_matches2 = (len(matches) / len(keypoints2)) * 100
    success = 0.5 * (success_of_matches1 + success_of_matches2)
    print('Success of matches: ' + str(success))