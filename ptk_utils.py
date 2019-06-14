"""
Provides an interface to the python tree kernel (ptk) package.

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
from tree_utils import check_tree_structure
from ptk.tree import Tree
from ptk.tree import TreeNode
import ptk.tree_kernels

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

def to_ptk_tree(nodes, adj):
    """ Converts a tree in node list/adjacency list format to a tree in
    Giovanni Di San Martino's TreeNode format.

    Args:
    nodes:   a list of nodes for the tree.
    adj:     an adjacency list for the tree.

    Returns: the tree in TreeNode format.
    """
    # verify that the given adjacency list corresponds to a tree
    par = check_tree_structure(adj)
    # retrieve the root of the tree
    r = np.where(par < 0)[0]
    if(len(r) > 1):
        raise ValueError("The input tree has multiple roots")
    # start converting from the root and then continue recursively
    root = _to_ptk_node(nodes, adj, r[0])
    return Tree(root)


def _to_ptk_node(nodes, adj, i):
    # retrieve the label or value for the current node
    val = nodes[i]
    # convert its children
    chs = []
    for j in adj[i]:
        chs.append(_to_ptk_node(nodes, adj, j))
    # return the TreeNode representing the tree up to i
    return TreeNode(val, chs)


def _kernel_with_indices(k, l, x, y, kernel):
    return (k, l, kernel(x, y))

def pairwise_kernel(Xs, Ys, kernel = 'ST', lambda_ = .75, num_jobs = 8):
    """ Computes the pairwise kernel values between the trees in
    Xs and the trees in Ys. Each entry of Xs and Ys is supposed to be in PTK
    format.

    Note that we assume a proper depth-first-search order of adj, i.e. for
    every node i, the following indices are all part of the subtree rooted at
    i until we hit the index of i's right sibling or the end of the tree.

    Args:
    Xs:       a list of trees in PTK format.
    Ys:       a list of trees in PTK format.
    kernel:   either ST (for subtree kernel), SST (for subset tree kernel), or
              PT (for partial tree kernel). Defaults to ST.
    lambda_:  the lambda hyper parameter for the subtree, subset tree, and
              partial tree kernel. Defaults to 1.0.
    num_jobs: The number of jobs to be used for parallel processing. Defaults
              to 8.

    Returns: a len(Xs) x len(Ys) matrix of pairwise kernel values.
    """
    # set up a parallel processing pool
    pool = mp.Pool(num_jobs)
    # set up the result matrix
    K = np.zeros((len(Xs),len(Ys)))

    # select the kernel
    if(kernel == 'ST'):
        kernel_obj = ptk.tree_kernels.KernelST(lambda_)
    elif(kernel == 'SST'):
        kernel_obj = ptk.tree_kernels.KernelSST(lambda_)
    elif(kernel == 'PT'):
        kernel_obj = ptk.tree_kernels.KernelSST(lambda_)
    kernel = kernel_obj.kernel

    # set up the callback function
    def callback(tpl):
        K[tpl[0], tpl[1]] = tpl[2]
    def error_callback(e):
        raise e

    # start off all parallel processing jobs
    for k in range(len(Xs)):
        for l in range(len(Ys)):
            pool.apply_async(_kernel_with_indices, args=(k, l, Xs[k], Ys[l], kernel), callback=callback, error_callback=error_callback)

    # wait for the jobs to finish
    pool.close()
    pool.join()

    # return the kernel matrix
    return K

def pairwise_kernel_symmetric(Xs, kernel = 'ST', lambda_ = .75, num_jobs = 8):
    """ Computes the pairwise kernel values between the trees in
    Xs. Each entry of Xs is supposed to be in PTK format.

    Note that this method is more efficient than pairiwse_kernels(Xs, Xs)
    because we only compute the upper triangle and then symmetrize.    pairwise_distances(Xs, Xs, delta, num_jobs).

    Args:
    Xs:      a list of trees in PTK format.
    kernel:   either ST (for subtree kernel), SST (for subset tree kernel), or
              PT (for partial tree kernel). Defaults to ST.
    lambda_:  the lambda hyper parameter for the subtree, subset tree, and
              partial tree kernel. Defaults to 1.0.
    num_jobs: The number of jobs to be used for parallel processing.
              Defaults to 8.

    Returns: a symmetric len(Xs) x len(Xs) matrix of pairwise kernel values.
    """
    # set up a parallel processing pool
    pool = mp.Pool(num_jobs)
    # set up the result matrix
    K = np.zeros((len(Xs), len(Xs)))

    # select the kernel
    if(kernel == 'ST'):
        kernel_obj = ptk.tree_kernels.KernelST(lambda_)
    elif(kernel == 'SST'):
        kernel_obj = ptk.tree_kernels.KernelSST(lambda_)
    elif(kernel == 'PT'):
        kernel_obj = ptk.tree_kernels.KernelSST(lambda_)
    kernel = kernel_obj.kernel

    # set up the callback function
    def callback(tpl):
        K[tpl[0], tpl[1]] = tpl[2]
    def error_callback(e):
        raise e

    # start off all parallel processing jobs
    for k in range(len(Xs)):
        for l in range(k, len(Xs)):
            pool.apply_async(_kernel_with_indices, args=(k, l, Xs[k], Xs[l], kernel), callback=callback, error_callback=error_callback)

    # wait for the jobs to finish
    pool.close()
    pool.join()

    # add the lower diagonal
    K += np.tril(K.T, -1)

    # return the kernel matrix
    return K
