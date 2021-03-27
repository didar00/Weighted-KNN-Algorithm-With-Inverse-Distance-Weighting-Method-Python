'''

            WEIGHTED K NEAREST NEIGHBOR ALGORITHM


'''


''' imports '''
from random import randrange
from math import sqrt
import numpy as np
import os



'''
Calculate the Euclidean distance
(sum of squared distances)
between two vectors of data
'''
def euclidean_distance(vec1, vec2):
    distance = 0.0
    for i in range(1, len(vec2)-1):
        distance += np.sum(np.square(vec1[i]-vec2[i]))
    return np.sqrt(distance)



'''
Cross validation method to split a data into n folds,
return grouped data inside a list
'''
def cross_validation(data, n):
    folds = list()
    data_copy = list(data)
    fold_size = int(len(data)/n)
    for _ in range(n):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(data_copy))
            fold.append(data_copy.pop(index)) # delete the data row
        folds.append(fold)                    # after selecting it
    return folds


'''
Runs knn algorithm through applying cross validation
on training data in case test dataset is not provided
'''
def weighted_knn_with_CV(dataset, n, k):
    folds = cross_validation(dataset, n)
    accuracy_list = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = weighted_knn(train_set, test_set, k)
        actual = [row[-1] for row in fold]
        accuracy = get_accuracy(actual, predicted)
        accuracy_list.append(accuracy)
    return accuracy_list


'''
Runs weighted knn algorithm with the provided test dataset
'''
def weighted_knn(train_set, test_set, k):
    scores = list()
    predicted = weighted_knn_helper(train_set, test_set, k)
    actual = [row[-1] for row in test_set]
    accuracy = get_accuracy(actual, predicted)
    scores.append(accuracy)
    return scores


'''
Weighted K Nearest Neighbor algorithm to classify
the image according to the k neighbors' weights
'''
def weighted_knn_helper(train, test, k):
    predictions = list()
    for row in test:
        output = predict_classification(train, row, k)
        predictions.append(output)
    return(predictions)


'''
Predict label of the image through using
inverse distance weighting method
'''
def predict_classification(train, test_row, k):
    # get k nearest neighbors
    neighbors = get_neighbors(train, test_row, k)

    '''
    label dictionary to store their associated weights
    these labels can be changed to any label name
    according to the data that is used
    '''
    labels = {"label1" : 0, "label2" : 0, "label3" : 0}

    '''
    for each points/data in neighbor,
    sum up their inverse distance
    '''
    for neighbor in neighbors:
        label = neighbor[0][-1]
        dist = neighbor[1]
        # do not divide with zero
        if dist != 0:
            labels[label] += 1/dist
        
    # return the most weighted label 
    # as classification
    label = max(labels.keys(), key=(lambda key: labels[key]))
    return label


'''
Return the list of nearest k neighbors of test_row
after obtaining and sorting the distances
between test data and all training data
'''
def get_neighbors(train, test_row, k):
    distances = list()
    for train_row in train:
        # calculate euclidean distance between vectors
        dist = euclidean_distance(test_row, train_row)
        # add distance to list to sort it later
        distances.append((train_row, dist))
    # sort distances in ascending order
    # to find the nearest neighbors
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(k):
        neighbors.append(distances[i])
    return neighbors


'''
Calculate the accuracy of classification
through calculating the actual and predicted
result
'''
def get_accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0