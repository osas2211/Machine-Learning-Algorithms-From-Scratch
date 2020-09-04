# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 23:33:22 2020

@author: OSARIEMEN OSARETIN FRANK
"""


import numpy as np
import pandas as pd
from collections import Counter
from warnings import warn

def euc_dist(x1,x2):
    sum_diff = []
    #when a pandas DataFrame is passed
    try:
        
        for i in range(0, len(x1.values)):
            x11,x22 = x1.values,x2
            for j in x1.values:
                for k in range(0,len(j)):
                    for p in range(0, len(x22)):
                        diff = x11[i][k] - x22[i][k]
                        sum_diff.append(diff**2)
        euclidean_distance =  np.sqrt(np.sum(sum_diff))        
        return  euclidean_distance
    
    except:
        for i in range(0,len(x1)):
                x11,x22 = x1,x2
                for j in x1:
                    for k in range(0, len(j)):
                        for p in range(0, len(x22)):
                            diff = x11[i][k] - x22[i][k]
                            sum_diff.append(diff**2)
        euclidean_distance = np.sqrt(np.sum(sum_diff))
        return euclidean_distance
        
    finally:
        return np.linalg.norm(x1 - x2)

class KNN:
    def __init__(self, k=5):
        self.k = k
    
    def store_values(self,X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        try:
            predicted_labels = [self._predict(x) for x in X.values]
            return np.array(predicted_labels)
        except:
            predicted_labels = [self._predict(x) for x in X]
            return np.array(predicted_labels)
        
    
    def _predict(self,x):
        try:
            distances = [euc_dist(x, x_train) for x_train in self.X_train]
            closest_distances = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in closest_distances]
            majority_vote = Counter(k_nearest_labels).most_common(1)[0][0]
            return majority_vote
        except:
            distances = [euc_dist(x, x_train) for x_train in self.X_train.values]
            closest_distances = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train.values[i] for i in closest_distances]
            majority_vote = Counter(k_nearest_labels).most_common(1)[0][0]
            return majority_vote
    
    def accuracy(self,y_test,pred):
        correct = 0
        total = len(pred)
        for i,j in zip(pred,y_test):
            if i == j:
                correct += 1
        accuracy = correct/total
        return accuracy
        
        
        