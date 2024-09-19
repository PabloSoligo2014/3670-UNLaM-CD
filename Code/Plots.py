import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import seaborn as sns
import pandas as pd
from matplotlib.collections import EllipseCollection
from matplotlib.colors import Normalize
from matplotlib.collections import EllipseCollection
from matplotlib.colors import Normalize


def plt_validation_curve(hiperparams, train_scores, valid_scores, scoring='accuracy', ylims = None, figsize=(8, 6)):
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    valid_mean = np.mean(valid_scores, 1)
    valid_std = np.std(valid_scores, 1)
    
    maxmax = max(np.max(valid_mean), np.max(train_mean))
    minmin = min(np.min(valid_mean), np.min(train_mean))
    plt.figure(figsize=figsize)
    plt.plot(hiperparams, train_mean, color='blue', marker='o', markersize=5, label='Training accuracy')
    plt.fill_between(hiperparams, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(hiperparams, valid_mean, color='green', linestyle='--', marker='s', markersize=5,label='Validation accuracy')
    plt.fill_between(hiperparams,valid_mean + valid_std, valid_mean - valid_std, alpha=0.15, color='blue')

    plt.grid()
    plt.legend(loc='best')
    plt.xlim(min(hiperparams), max(hiperparams))
    plt.xlabel('Param')
    plt.ylabel('score')
    plt.legend(loc='upper right')
    plt.title("Scoring {}, value: {:.2}, param:{}".format(scoring, np.mean(valid_scores, 1).max(), hiperparams[np.mean(valid_scores, 1).argmax()]))
    #Atencion, es el maximo de todos los scores, no es representativo del score real

    if ylims is not None:
        plt.ylim(ylims)
    else:
        plt.ylim([minmin-(minmin*0.009), maxmax+(maxmax*0.009)])
    plt.show() 

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



def plot_corr_ellipses(data, figsize=None, **kwargs):
    ''' https://stackoverflow.com/a/34558488 '''
    M = np.array(data)
    if not M.ndim == 2:
        raise ValueError('data must be a 2D array')
    fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw={'aspect':'equal'})
    ax.set_xlim(-0.5, M.shape[1] - 0.5)
    ax.set_ylim(-0.5, M.shape[0] - 0.5)
    ax.invert_yaxis()

    # xy locations of each ellipse center
    xy = np.indices(M.shape)[::-1].reshape(2, -1).T

    # set the relative sizes of the major/minor axes according to the strength of
    # the positive/negative correlation
    w = np.ones_like(M).ravel() + 0.01
    h = 1 - np.abs(M).ravel() - 0.01
    a = 45 * np.sign(M).ravel()

    ec = EllipseCollection(widths=w, heights=h, angles=a, units='x', offsets=xy,
                           norm=Normalize(vmin=-1, vmax=1),
                           transOffset=ax.transData, array=M.ravel(), **kwargs)
    ax.add_collection(ec)

    # if data is a DataFrame, use the row/column names as tick labels
    if isinstance(data, pd.DataFrame):
        ax.set_xticks(np.arange(M.shape[1]))
        ax.set_xticklabels(data.columns, rotation=90)
        ax.set_yticks(np.arange(M.shape[0]))
        ax.set_yticklabels(data.index)

    return ec, ax

