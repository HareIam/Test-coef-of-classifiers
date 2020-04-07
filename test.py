import matplotlib.pyplot as plt
from sklearn import svm
import warnings
import warnings

import matplotlib.pyplot as plt
from sklearn import svm

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def get_model(name):
    '''
    This function is used to return classifier model with a classifier name as input
    '''

    random_seed_num=21
    if name == 'Logistic Regression':
        return LogisticRegression(solver='liblinear',max_iter=100000,random_state=random_seed_num,class_weight="balanced")
    elif name == 'SVM':
        return SVC(kernel='linear', probability=True, max_iter=100000,random_state=random_seed_num,class_weight="balanced")
    elif name == 'Decision tree':
        return tree.DecisionTreeClassifier(random_state=random_seed_num,class_weight="balanced")
    elif name == 'SVC':
        return SVC(kernel='poly', probability=True, max_iter=100000,random_state=random_seed_num,class_weight="balanced")
    elif name == 'SVM_rbf':
        return SVC(kernel='rbf', probability=True, max_iter=100000,random_state=random_seed_num,class_weight="balanced")
    elif name == 'MultinomialNB':
        return MultinomialNB()
    elif name == 'Gradient Boosting':
        return GradientBoostingClassifier(n_estimators=400, random_state=random_seed_num)
    elif name == 'KNeighborsClassifier':
        return KNeighborsClassifier(3)
    elif name == 'MLP':
        return MLPClassifier(solver='lbfgs', max_iter=10000,random_state=random_seed_num)
    elif name == 'NaiveBayes':
        return GaussianNB()
    elif name == "AdaBoost":
        return AdaBoostClassifier(n_estimators=400,random_state=random_seed_num)
    elif name == "RandomForest":
        return RandomForestClassifier(max_depth=7, n_estimators=400,random_state=random_seed_num,class_weight="balanced")
    else:
        raise ValueError('No such model')



def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

if __name__ == '__main__':

    from sklearn import datasets
    # X=np.array([[0,1],[0,2],[1,2],[0.5,1],[1,0],[2,0],[2,1],[1.5,1]])
    # y=np.array([0,0,0,0,1,1,1,1])

    # we create 40 separable points
    np.random.seed(0)
    X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
    y = [0] * 20 + [1] * 20

    Classifier_list = ['Logistic Regression', 'SVM', 'Gradient Boosting', 'AdaBoost', 'RandomForest']
    fn='Gradient Boosting'
    clf = get_model(fn)
    clf.fit(X, y)
    print(clf.coef_)

    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    C = 1.0  # SVM regularization parameter
    models = [get_model(fn)]
    models = [clf.fit(X, y) for clf in models]

    # title for the plots
    titles = [fn]

    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(1, 1)
    # plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    for clf, title, ax in zip(models, titles, [sub]):
        plot_contours(ax, clf, xx, yy,
                      cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)

    plt.savefig('./'+fn+'.png',dpi=300)
    plt.show()