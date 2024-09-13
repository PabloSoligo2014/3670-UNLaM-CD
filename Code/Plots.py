import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import seaborn as sns


def plt_learning_curve(train_sizes, train_scores, test_scores, scoring, ylims = None, figsize=(8, 6)):
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    maxmax = max(np.max(test_mean), np.max(train_mean))
    minmin = min(np.min(test_mean), np.min(train_mean))
    
    plt.figure(figsize=figsize)
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training accuracy')
    plt.fill_between(train_sizes,train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5,label='Validation accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    plt.grid()
    plt.xlabel('Number of training examples')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')
    plt.title("Scoring :"+scoring)
    if ylims is not None:
        plt.ylim(ylims)
    else:
        plt.ylim([minmin-(minmin*0.009), maxmax+(maxmax*0.009)])
    plt.show()   

