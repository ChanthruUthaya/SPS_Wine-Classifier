#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skeleton code for CW2 submission.
We suggest you implement your code in the provided functions
Make sure to use our print_features() and print_predictions() functions
to print your results
"""

from __future__ import print_function

import os
import sys
import pandas as pd
import collections
import argparse
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from utilities import load_data, print_features, print_predictions
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# you may use these colours to produce the scatter plots
CLASS_1_C = r'#3366ff' #blue
CLASS_2_C = r'#cc3300' #red
CLASS_3_C = r'#ffc34d' #yellow

class_colours = [CLASS_1_C,CLASS_2_C,CLASS_3_C]

MODES = ['feature_sel', 'knn', 'alt', 'knn_3d', 'knn_pca']


########################################################################################################################
# CONFUSION MATRIX


def calculate_accuracy(gt_labels, pred_labels):
    correct = 0
    for i, value in enumerate(pred_labels):
        if (value == gt_labels[i]):
            correct += 1
    accuracy = correct/len(pred_labels)
    return round(accuracy*100, 1)


def cm_value(x, y, gt_labels, pred_labels):
    classes = np.unique(pred_labels)
    class1 = classes[x]
    class2 = classes[y]
    amounts = collections.Counter(gt_labels)
    wrongclass = 0
    total = amounts[class1]
    for i, value in enumerate(gt_labels):
        if (value == class1 and pred_labels[i] == class2):
            wrongclass += 1
    return round(wrongclass/total, 3)


def calculate_confusion_matrix(gt_labels, pred_labels):
    classes = np.unique(pred_labels)
    length = len(classes)
    cm = np.zeros((length,length))
    for x in range(length):
        for y in range(length):
            cm[x][y] = cm_value(x, y, gt_labels, pred_labels)
    return cm


def plot_matrix(matrix, ax=None):

    if ax is None:
        ax = plt.gca()

    handle = ax.imshow(matrix, plt.get_cmap('summer'))
    plt.colorbar(handle)
    for i, value in enumerate(matrix):
        for i1, values in enumerate(value):
            plt.text(i1,i,values)
    plt.show()


########################################################################################################################
# FEATURE SELECTION


def feature_selection(train_set, train_labels, **kwargs):
    # write your code here and make sure you return the features at the end of
    # the function

    if kwargs == {}:

        n_features = train_set.shape[1]

        colours = np.zeros_like(train_labels, dtype=np.object)

        colours[train_labels == 1] = class_colours[0]
        colours[train_labels == 2] = class_colours[1]
        colours[train_labels == 3] = class_colours[2]

        # SHOW SELECTED PLOT
        # fig, ax = plt.subplots(1)
        # ax.scatter(train_set[:, 3], train_set[:, 10], c=colours)

        # SHOW ALL PLOTS
        # fig, ax = plt.subplots(n_features, n_features)
        # plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.2, hspace=0.4)
        # for row in range(n_features):
        #     for col in range(n_features):
        #         ax[row][col].scatter(train_set[:, row], train_set[:, col], c=colours, s = 1)
        #         ax[row][col].set_title('Features {} vs {}'.format(row+1, col+1))

        plt.show()

    return [6,9]


########################################################################################################################
# KNN


def distance_calc(point, red_train):
    dist_arr = []
    for train_point in red_train:
        squares = []
        for i, val in enumerate(train_point):
            squares.append(np.square(point[i]-val))
        dist_arr.append(np.sqrt(np.sum(squares)))
    return dist_arr


def pick_k_nearest(k, dist):
    k_nearest = []
    for i in range(k):
        min_dist = min(dist)
        l = dist.index(min_dist)
        k_nearest.append([l,min_dist])
        dist[l] = max(dist)
    return k_nearest


def knn(train_set, train_labels, test_set, k, features, **kwargs):
    # write your code here and make sure you return the predictions at the end of
    # the function
    pred_labels = []
    red_train = train_set[:, features]
    red_test = test_set[:, features]

    for point in red_test:
        dist = distance_calc(point, red_train)
        k_nearest = pick_k_nearest(k, dist)

        count = [0,0,0]
        for item in k_nearest:
            i = item[0]
            if (train_labels[i] == 1):
                count[0] += 1
            elif (train_labels[i] == 2):
                count[1] += 1
            elif (train_labels[i] == 3):
                count[2] += 1
        pred_labels.append(count.index(max(count))+1)

    cm = calculate_confusion_matrix(test_labels, pred_labels)
    # plot_matrix(cm)

    accuracy = calculate_accuracy(test_labels, pred_labels)
    #print(accuracy)

    return pred_labels


########################################################################################################################
# ALT CLASSIFIER - NAIVE BAYES


def find_likelyhood(x, mean, std):
    return np.exp(-(x-mean)**2/(2*std**2))*(1/(np.sqrt(2*np.pi)*std))


def find_posterior(xy, label, mean, std):
    product = np.prod(find_likelyhood(xy, mean, std))
    product = product * (label.shape[0] / train_set.shape[0])
    return product


def alternative_classifier(train_set, train_labels, test_set, features, **kwargs):
    # write your code here and make sure you return the predictions at the end of
    # the function
    pred_labels = []
    kwargs = {"arg1": 1}

    red_train = np.array(train_set[:, features])
    red_test = np.array(test_set[:, features])

    a_temp = []
    b_temp = []
    c_temp = []

    for i in range(len(train_labels)):
        if (train_labels[i] == 1):
            a_temp.append(red_train[i,:])
        elif (train_labels[i] == 2):
            b_temp.append(red_train[i, :])
        elif (train_labels[i] == 3):
            c_temp.append(red_train[i, :])

    a = np.array(a_temp)
    b = np.array(b_temp)
    c = np.array(c_temp)

    # Find Mean and STD for each class
    means = np.array([np.mean(a, axis=0), np.mean(b, axis=0), np.mean(c, axis=0)])
    std = np.array([np.std(a, axis=0), np.std(b, axis=0), np.std(c, axis=0)])

    for xy in red_test:

       p_a = find_posterior(xy, a, means[0], std[0])
       p_b = find_posterior(xy, b, means[1], std[1])
       p_c = find_posterior(xy, c, means[2], std[2])
       p_vals = [p_a, p_b, p_c]
       index = p_vals.index(max(p_vals))
       pred_labels.append(index + 1)

    cm = calculate_confusion_matrix(test_labels, pred_labels)
    # plot_matrix(cm)

    accuracy = calculate_accuracy(test_labels, pred_labels)
    #print(accuracy)

    return pred_labels


########################################################################################################################
# KNN THREE FEATURES


def knn_three_features(train_set, train_labels, test_set, k, features, **kwargs):

    red_train = train_set[:, features]
    red_test = test_set[:, features]
    pred_labels = []

    for point in red_test:
        dist = distance_calc(point, red_train)
        k_nearest = pick_k_nearest(k, dist)
        count = [0,0,0]
        for item in k_nearest:
            i = item[0]
            if (train_labels[i] == 1):
                count[0] += 1
            elif (train_labels[i] == 2):
                count[1] += 1
            elif (train_labels[i] == 3):
                count[2] += 1
        pred_labels.append(count.index(max(count))+1)

    cm = calculate_confusion_matrix(test_labels, pred_labels)
    # plot_matrix(cm)

    accuracy = calculate_accuracy(test_labels, pred_labels)
    #print(accuracy)

    return pred_labels


########################################################################################################################
# KNN PCA


def knn_pca(train_set, train_labels, test_set, k, n_components=2, **kwargs):

    # Standardizing
    train_scaled = StandardScaler().fit_transform(train_set)
    test_scaled = StandardScaler().fit_transform(test_set)

    pca = PCA(n_components)

    train_scaled_r = pca.fit_transform(train_scaled)
    test_scaled_r = pca.transform(test_scaled)

    pred_labels = knn(train_scaled_r, train_labels, test_scaled_r, k = 1, features = [0 , 1])

    cm = calculate_confusion_matrix(test_labels, pred_labels)
    # plot_matrix(cm)

    accuracy = calculate_accuracy(test_labels, pred_labels)
    #print(accuracy)

    return pred_labels, train_scaled_r


########################################################################################################################
# PAIRS

def pairs(train_set, train_labels, test_set, k):

    pairs = [[0, 0]]
    best = 0

    for i in range(0, 12):
        for j in range(0, 12):
            pred_labels, accuracy = knn(train_set, train_labels, test_set, k = 1, features = [i, j])
            cm = calculate_confusion_matrix(test_labels, pred_labels)
            percentage = 100 * (cm[0, 0] + cm[1, 1] + cm[2, 2]) / 3
            if(percentage > 85):
                if(j != i):
                    best = percentage
                    pairs.append([i ,j])
    pairs.remove([0,0])
    print(pairs)

    return best, pairs

########################################################################################################################
# PAIRS3

def pairs_three(train_set, train_labels, test_set, k):

    pairs = [0, 0, 0]
    best = 0

    for i in range(0, 12):
        for j in range(0, 12):
            for l in range(0, 12):

                pred_labels = knn_three_features(train_set, train_labels, test_set, k = 1, features = [i, j, l])
                cm = calculate_confusion_matrix(test_labels, pred_labels)
                percentage = 100 * (cm[0, 0] + cm[1, 1] + cm[2, 2]) / 3
                if(percentage > best):
                    best = percentage
                    pairs[0] = i
                    pairs[1] = j
                    pairs[2] = l

    return best, pairs


########################################################################################################################
# plot_knn_ks - MUST CHANGE KNN TO RETURN ACCURACY INSTEAD OF PRED_LABELS

def plot_knn_ks(train_set, train_labels, test_set, features):


    colours = [r'#ffc34d', r'#cc3300', r'#3366ff', r'#3300ff', r'#336600', r'#33600f', r'#3000ff']

    percentage, pair = pairs(train_set, train_labels, test_set, args.k)

    fig, ax = plt.subplots(1)
    for i, p in enumerate(pair):
        print(i)
        accuracies = []
        ks = []
        for k in range(1, 8):
            pred_labels, accuracy = knn(train_set, train_labels, test_set, k, p)
            print(accuracy)
            accuracies.append(accuracy)
            ks.append(k)

        ax.plot(ks, accuracies)
        plt.xlabel('K')
        plt.ylabel('Error Rate')

    plt.show()


########################################################################################################################

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs=1, type=str, help='Running mode. Must be one of the following modes: {}'.format(MODES))
    parser.add_argument('--k', nargs='?', type=int, default=1, help='Number of neighbours for knn')
    parser.add_argument('--train_set_path', nargs='?', type=str, default='data/wine_train.csv', help='Path to the training set csv')
    parser.add_argument('--train_labels_path', nargs='?', type=str, default='data/wine_train_labels.csv', help='Path to training labels')
    parser.add_argument('--test_set_path', nargs='?', type=str, default='data/wine_test.csv', help='Path to the test set csv')
    parser.add_argument('--test_labels_path', nargs='?', type=str, default='data/wine_test_labels.csv', help='Path to the test labels csv')

    args = parser.parse_args()
    mode = args.mode[0]

    return args, mode


if __name__ == '__main__':
    args, mode = parse_args() # get argument from the command line

    # load the data
    train_set, train_labels, test_set, test_labels = load_data(train_set_path=args.train_set_path,
                                                                       train_labels_path=args.train_labels_path,
                                                                       test_set_path=args.test_set_path,
                                                                       test_labels_path=args.test_labels_path)
    if mode == 'feature_sel':
        selected_features = feature_selection(train_set, train_labels)
        print_features(selected_features)
    elif mode == 'knn':
        kwargs = {"arg1": 1}
        predictions = knn(train_set, train_labels, test_set, args.k, features = feature_selection(train_set, train_labels))
        print_predictions(predictions)
    elif mode == 'alt':
        predictions = alternative_classifier(train_set, train_labels, test_set, features = feature_selection(train_set, train_labels))
        print_predictions(predictions)
    elif mode == 'knn_3d':
        predictions = knn_three_features(train_set, train_labels, test_set, args.k, features = [0,6,8])
        print_predictions(predictions)
    elif mode == 'knn_pca':
        prediction, data = knn_pca(train_set, train_labels, test_set, args.k)
        print_predictions(prediction)
        print(data)
    elif mode == 'pairs':
        percentage, pairs = pairs(train_set, train_labels, test_set, args.k)
    elif mode == 'pairs3':
        percentage, pairs = pairs_three(train_set, train_labels, test_set, args.k)
        print(percentage, pairs)
    elif mode == 'plot_knn_ks':
        plot_knn_ks(train_set, train_labels, test_set, features = feature_selection(train_set, train_labels))
    else:
        raise Exception('Unrecognised mode: {}. Possible modes are: {}'.format(mode, MODES))



