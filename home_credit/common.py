# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import re

def data_load_home_credit(path):
    """
    To load dataset from the directory for home credit and return test, train DataFrame
    """
    data = pd.read_csv('/media/ismaeel/Work/msds19029_thesis/dataset/home_with_missing.csv')
    data = data.dropna(axis=0, subset=['TARGET'])

    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    feature_name = missing_data[missing_data['Percent'] > 10]
    data.drop(list(feature_name.index), axis=1, inplace=True)

    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    feature_name = missing_data[missing_data['Percent'] != 0]
    data = data.dropna(axis=0, subset=list(feature_name.index))

    data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

    Y = data['TARGET']
    X = data
    train, test, __, __ = train_test_split(X, Y, test_size=0.10, random_state=42)

    return train, test

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

def find_optimal_cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations
    predicted : Matrix with predicted data, where rows are observations
    Returns
    -------     
    list type, with optimal cutoff value
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    return list(roc_t['threshold']) 

def plot_auc(labels, pred):
    """
    Plot auc plot against the labels and pred
    """
    ns_probs = [0 for _ in range(len(labels))]
    ns_auc = roc_auc_score(labels, ns_probs)
    lr_auc = roc_auc_score(labels, pred)
    print('Binary: ROC AUC=%.3f' % (lr_auc))
    plt.figure(figsize=(5, 5))
    ns_fpr, ns_tpr, th1 = roc_curve(labels, ns_probs)
    lr_fpr, lr_tpr, th2 = roc_curve(labels, pred)
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Binary Classification')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show