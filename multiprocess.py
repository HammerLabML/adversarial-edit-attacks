"""
Implements parallel computations of tree edit distances.

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
import ted

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

def _standard_ted_with_indices(k, l, x, x_adj, y, y_adj):
    return (k, l, ted.standard_ted(x, x_adj, y, y_adj))

def _ted_with_indices(k, l, x, x_adj, y, y_adj, delta):
    return (k, l, ted.ted(x, x_adj, y, y_adj, delta))

def pairwise_distances(Xs, Ys, delta = None, num_jobs = 8):
    """ Computes the pairwise tree edit distances between the trees in
    Xs and the trees in Ys. Each entry of Xs and Ys is supposed to be a
    tuple of a node list and an adjacency list adj, where adj[i]
    is a list of indices pointing to children of node i.

    Note that we assume a proper depth-first-search order of adj, i.e. for
    every node i, the following indices are all part of the subtree rooted at
    i until we hit the index of i's right sibling or the end of the tree.

    Args:
    Xs:      a list of trees, where each tree is a tuple of a node list and
             an adjacency list
    Ys:      a list of trees, where each tree is a tuple of a node list and
             an adjacency list
    delta:   a function that takes two nodes as inputs and returns their
             pairwise distance, where delta(x, None) should be the cost of
             deleting x and delta(None, y) should be the cost of inserting y.
             If undefined, this method calls standard_ted instead.
    num_jobs: The number of jobs to be used for parallel processing.

    Returns: a len(Xs) x len(Ys) matrix of pairwise tree edit distance values.
    """
    K = len(Xs)
    L = len(Ys)
    # set up a parallel processing pool
    pool = mp.Pool(num_jobs)
    # set up the result matrix
    if(delta is None):
        D = np.zeros((K,L), dtype=int)
    else:
        D = np.zeros((K,L))

    # set up the callback function
    def callback(tpl):
        D[tpl[0], tpl[1]] = tpl[2]
    def error_callback(e):
        raise e

    # start off all parallel processing jobs
    if(delta is None):
        for k in range(K):
            for l in range(L):
                pool.apply_async(_standard_ted_with_indices, args=(k, l, Xs[k][0], Xs[k][1], Ys[l][0], Ys[l][1]), callback=callback, error_callback=error_callback)
    else:
        for k in range(K):
            for l in range(L):
                pool.apply_async(_ted_with_indices, args=(k, l, Xs[k][0], Xs[k][1], Ys[l][0], Ys[l][1], delta), callback=callback, error_callback=error_callback)

    # wait for the jobs to finish
    pool.close()
    pool.join()

    # return the distance matrix
    return D

def pairwise_distances_symmetric(Xs, delta = None, num_jobs = 8):
    """ Computes the pairwise tree edit distances between the trees in
    Xs. Each entry of Xs is supposed to be a tuple of a node list
    and an adjacency list adj, where adj[i] is a list of indices pointing
    to children of node i.

    Note that we assume a proper depth-first-search order of adj, i.e. for
    every node i, the following indices are all part of the subtree rooted at
    i until we hit the index of i's right sibling or the end of the tree.

    Further note that this method assumes that delta is a self-identical and
    symmetric function. Therefore, only the upper triangle of the pairwise
    distance matrix is computed and then mirrored to the bottom diagonal.
    Therefore, this method is about twice as fast as

    pairwise_distances(Xs, Xs, delta, num_jobs).

    Args:
    Xs:      a list of trees, where each tree is a tuple of a node list and
             an adjacency list
    delta:   a function that takes two nodes as inputs and returns their
             pairwise distance, where delta(x, None) should be the cost of
             deleting x and delta(None, y) should be the cost of inserting y.
             If undefined, this method calls standard_ted instead.
    num_jobs: The number of jobs to be used for parallel processing.

    Returns: a symmetric len(Xs) x len(Xs) matrix of pairwise tree edit
             distance values.
    """
    K = len(Xs)
    # set up a parallel processing pool
    pool = mp.Pool(num_jobs)
    # set up the result matrix
    if(delta is None):
        D = np.zeros((K,K), dtype=int)
    else:
        D = np.zeros((K,K))

    # set up the callback function
    def callback(tpl):
        D[tpl[0], tpl[1]] = tpl[2]
    def error_callback(e):
        raise e

    # start off all parallel processing jobs
    if(delta is None):
        for k in range(K):
            for l in range(k+1, K):
                pool.apply_async(_standard_ted_with_indices, args=(k, l, Xs[k][0], Xs[k][1], Xs[l][0], Xs[l][1]), callback=callback, error_callback=error_callback)
    else:
        for k in range(K):
            for l in range(k+1, K):
                pool.apply_async(_ted_with_indices, args=(k, l, Xs[k][0], Xs[k][1], Xs[l][0], Xs[l][1], delta), callback=callback, error_callback=error_callback)

    # wait for the jobs to finish
    pool.close()
    pool.join()

    # add the lower diagonal
    D += np.transpose(D)

    # return the distance matrix
    return D
