"""
Implements hyperparameter optimization for SVM with custom kernels and
tree echo state networks.

Copyright (C) 2019
Benjamin Paaßen
AG Machine Learning
Bielefeld University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import multiprocessing as mp
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tree_echo_state import TreeEchoStateNetClassifier

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

def _train_svm(f, c, k, K_train, Y_train, K_test, Y_test, C):
    # train an SVM for the current C and kernel matrix
    svm = SVC(kernel='precomputed', decision_function_shape='ovo', C = C)
    svm.fit(K_train, Y_train)
    # evaluate the test accuracy and return it
    Y_pred = svm.predict(K_test)
    return f, c, k, accuracy_score(Y_test, Y_pred)

def grid_search_cv_svm(Y, Cs, kernels, n_splits = 5, hyper_index = None, n_jobs = 8):
    """ Performs a hyper-parameter optimization for SVM for the given
    training data via crossvalidation.

    Args:
    Y:            An array of labels for the training data.
    Cs:           An array or list of possible C values which we should try.
    kernels :     A list of kernel matrices for all possible kernel hyper parameters.
    n_splits:     The number of crossvalidation folds for hyper-parameter
                  optimization. Defaults to 5.
    hyper_index:  The data point indices on which the hyper-parameter selection
                  should be done. Defaults to every point.
    n_jobs:       The number of jobs for parallel processing. Defaults to 8.

    Returns:
    C:    The optimal C parameter.
    k:    The index of the optimal kernel matrix.
    accs: A n_splits x len(Cs) x len(kernel_param) matrix of test accuracies
          which was used to select C and K.
    """
    # handle the case with no hyper parameter indices.
    # In this case, we just take all the points for training.
    if(hyper_index is None):
        hyper_index = np.arange(len(Y), dtype=int)

    # set up accuracy matrix
    accs = np.zeros((n_splits, len(Cs), len(kernels)))

    # set up a parallel processing pool
    pool = mp.Pool(n_jobs)

    # set up callback function
    def callback(tpl):
        f = tpl[0]
        c = tpl[1]
        k = tpl[2]
        acc = tpl[3]
        accs[f, c, k] = acc
    def error_callback(e):
        raise e

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
    # iterate over training and test splits
    f = 0
    for train_index2, test_index2 in cv.split(Y[hyper_index], Y[hyper_index]):
        train_index = hyper_index[train_index2]
        test_index  = hyper_index[test_index2]
        # iterate over all possible kernel parameter values
        for k in range(len(kernels)):
            # retrieve the kernel matrix for the current parameter
            K_train = kernels[k][train_index, :][:, train_index]
            K_test  = kernels[k][test_index, :][:, train_index]
            # iterate over all possible Cs
            for c in range(len(Cs)):
                # start parallel processing job for this crossvalidation fold
                pool.apply_async(_train_svm, args=(f, c, k, K_train, Y[train_index], K_test, Y[test_index], Cs[c]), callback=callback, error_callback=error_callback)
        f += 1

    # wait for the jobs to finish
    pool.close()
    pool.join()

    # average over the folds
    accs_mean = np.mean(accs, axis=0)
    # take the best hyper parameters
    [c_opt, k] = np.unravel_index(np.argmax(accs_mean), accs_mean.shape)
    C = Cs[c_opt]
    return C, k, accs

# for hyper-parameter optimization on echo state networks we need specialized
# multi-processing classes because, per default, multi-processing inside
# a multi-processing pool is not permitted
# This workaround is due to:
# https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
# and
# https://stackoverflow.com/questions/52948447/error-group-argument-must-be-none-for-now-in-multiprocessing-pool

class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass

class NoDaemonContext(type(mp.get_context())):
    Process = NoDaemonProcess

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NoDaemonPool(mp.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(NoDaemonPool, self).__init__(*args, **kwargs)

def _select_from_list(X, idx):
    X_selec = []
    for i in idx:
        X_selec.append(X[i])
    return X_selec

def _train_echo_state(f, d, r, alphabet, X_train, Y_train, X_test, Y_test, dim, radius):
    # train a tree echo state network for the current parameter values
    rec_net = TreeEchoStateNetClassifier(dim, alphabet, radius=radius)
    rec_net.fit(X_train, Y_train)
    # evaluate the test accuracy and return it
    Y_pred = rec_net.predict(X_test)
    return f, d, r, accuracy_score(Y_test, Y_pred)

def grid_search_cv_echo_state(X, Y, alphabet, dims, radii, n_splits = 5, hyper_index = None, n_jobs = 8):
    """ Performs a hyper-parameter optimization for tree echo state networks
    for the given training data via crossvalidation.

    Args:
    X:            A list of training data trees in node list/adjacency list
                  format.
    Y:            An array of labels for the training data.
    alphabet:     A list of unique symbols that may occur in the data set.
    dims:         An array or list of neuron numbers we should try.
    radii:        An array or list of scaling factors we should try.
    n_splits:     The number of crossvalidation folds for hyper-parameter
                  optimization. Defaults to 5.
    hyper_index:  The data point indices on which the hyper-parameter selection
                  should be done. Defaults to every point.
    n_jobs:       The number of jobs for parallel processing. Defaults to 8.

    Returns:
    dim:    The optimal number of neurons.
    radius: The optimal scaling factor.
    accs:   A n_splits x len(dims) x len(radii) matrix of test accuracies
            which was used to select dim and radius.
    """
    # handle the case with no hyper parameter indices.
    # In this case, we just take all the points for training.
    if(hyper_index is None):
        hyper_index = np.arange(len(Y), dtype=int)

    # set up accuracy matrix
    accs = np.zeros((n_splits, len(dims), len(radii)))

    # set up a parallel processing pool
    pool = NoDaemonPool(processes=n_jobs)

    # set up callback function
    def callback(tpl):
        f = tpl[0]
        d = tpl[1]
        r = tpl[2]
        acc = tpl[3]
        accs[f, d, r] = acc
    def error_callback(e):
        raise e

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
    # iterate over training and test splits
    f = 0
    for train_index2, test_index2 in cv.split(Y[hyper_index], Y[hyper_index]):
        train_index = hyper_index[train_index2]
        test_index  = hyper_index[test_index2]
        # retrieve the training and test trees for the current fold
        X_train = _select_from_list(X, train_index)
        X_test  = _select_from_list(X, test_index)
        # iterate over all possible dims
        for d in range(len(dims)):
            # iterate over all possible radii
            for r in range(len(radii)):
                # start parallel processing job for this crossvalidation fold
                pool.apply_async(_train_echo_state, args=(f, d, r, alphabet, X_train, Y[train_index], X_test, Y[test_index], dims[d], radii[r]), callback=callback, error_callback=error_callback)
        f += 1

    # wait for the jobs to finish
    pool.close()
    pool.join()

    # average over the folds
    accs_mean = np.mean(accs, axis=0)
    # take the best hyper parameters
    [d_opt, r_opt] = np.unravel_index(np.argmax(accs_mean), accs_mean.shape)
    dim = dims[d_opt]
    radius = radii[r_opt]
    return dim, radius, accs
