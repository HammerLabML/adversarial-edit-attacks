"""
Implements tree echo state networks.

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
from tree_utils import root

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

class TreeEchoStateNet:
    """ A tree echo state network implementation, following the paper of
    Gallicchio and Micheli (2013, doi:10.1016/j.neucom.2012.08.017).

    In contrast to vanilla recursive neural networks, tree echo state
    networks are only trained via their output. In other words, we initialize
    the recursive connections to encode the tree at random and leave them
    unchanged an train only the output layer.

    Another noteable characteristic is that we consider non-positional
    recursive neural networks, i.e. we do not consider the order of children
    but just compute the mean of their encodings as input for the parent.
    This way, we avoid having to specify the exact number of children for
    each possible symbol.

    Attributes:
    _dim:       The encoding dimensionality.
    _alphabet:  A dictionary mapping alphabet symbols to indices.
    _Ws:        A K x _dim x _dim tensor containing the weight matrices
                for each symbol, where K is the alphabet size.
    _bs:        A K x _dim matrix containing the bias vectors
                for each symbol, where K is the alphabet size.
    _W_out:     The output weight matrix.
    _b_out:     The output bias.
    _regul:     The L2 regularization strength.
    _n_jobs:    The number of jobs for parallel encoding.
    """
    def __init__(self, dim, alphabet, sparsity = 0.1, radius = 0.9, regul = 1E-5, n_jobs = 8):
        """ Initializes a tree echo state network.

        Args:
        dim:      The encoding dimensionality
        alphabet: A list or set of unique symbols which may occur in input
                  trees.
        sparsity: The proportion of nonzero entries in the weight matrices.
                  Defaults to 0.1.
        radius:   The scaling factor for the weight matrices to ensure a
                  contractive mapping.
        regul:     The L2 regularization strength. Defaults to 1E-5.
        n_jobs:    The number of jobs for parallel encoding. Defaults to 8.
        """
        self._dim = dim
        # set up the symbol to index mapping
        self._alphabet = {}
        k = 0
        for sym in alphabet:
            self._alphabet[sym] = k
            k += 1
        # set up the recurrent connections
        self._Ws = np.zeros((len(alphabet), dim, dim))
        nonzeros = max(1, int(np.floor(sparsity * dim)))
        # initialize the connections for each symbol row by row.
        for k in range(len(alphabet)):
            for i in range(dim):
                # select the nonzero positions at random
                pos = np.random.randint(dim, size=nonzeros)
                # select signs at random
                signs = np.sign(np.random.rand(nonzeros) * 2. - 1.)
                # set the ith row of the recursive matrix for symbol k
                self._Ws[k, i, pos] = signs * radius / nonzeros
        # initialize the bias vectors at random in the interval [-radius/dim, radius/dim]
        self._bs = (np.random.rand(len(alphabet), dim) * 2. - 1.)*radius/dim
        # store the regularization strength
        self._regul = regul
        # ... and the number of jobs
        self._n_jobs = n_jobs

    def encode(self, x_nodes, x_adj):
        """ Computes the dim-dimensional encoding of the input tree.

        Args:
        x_nodes: The node list of the input tree.
        x_adj:   The adjacency list of the input tree.

        Returns: a self._dim dimensional vector.
        """
        # identify the root
        r = root(x_adj)
        # initialize an encoding matrix for all tree nodes
        encodings = np.zeros((len(x_nodes), self._dim))
        has_encoding = np.zeros(len(x_nodes), dtype=bool)
        # then, we use a stack to move recursively through the tree and
        # embed all nodes
        stk = [r]
        while(stk):
            # pop the most recent node from the stack
            i = stk.pop()
            sym_i = x_nodes[i]
            s = self._alphabet[sym_i]
            # handle the special case of leaves
            if(not x_adj[i]):
                # if i is a leaf, we can embed it using the layer with zero
                # input
                b = self._bs[s, :]
                encodings[i, :] = np.tanh(b)
                has_encoding[i] = True
            elif(np.all(has_encoding[x_adj[i]])):
                # if this is not a leaf but all children are already encoded,
                # we can add the encodings of all children and then use this
                # as input for the current symbol encoding
                h = np.mean(encodings[x_adj[i], :], axis=0)
                W = self._Ws[s, :,:]
                b = self._bs[s, :]
                encodings[i, :] = np.tanh(np.dot(W, h) + b)
                has_encoding[i] = True
            else:
                # if the children are not yet embedded, we first push i back
                # on the stack and then all its children
                stk.append(i)
                for j in x_adj[i]:
                    stk.append(j)
        # after this while loop is completed, all nodes are embedded and we
        # can return the encoding of the root node
        return encodings[r, :]

    def _encode_with_index(self, i, x_nodes, x_adj):
        return i, self.encode(x_nodes, x_adj)

    def encode_list(self, X):
        """ Encodes a list of trees in paralell.

        Args:
        X: A list of m trees in node list/adjacency list format.

        Returns: A m x self._dim matrix of encodings.
        """
        # set up an encoding matrix for all trees
        H = np.zeros((len(X), self._dim))
        # set up a parallel processing pool
        pool = mp.Pool(self._n_jobs)
        # set up the callback functions
        def callback(tpl):
            H[tpl[0], :] = tpl[1]
        def error_callback(e):
            raise e
        # start off all parallel processing jobs
        for i in range(len(X)):
            pool.apply_async(self._encode_with_index, args=(i, X[i][0], X[i][1]), callback=callback, error_callback=error_callback)

        # wait for the jobs to finish
        pool.close()
        pool.join()
        # return the encoding matrix
        return H

    def fit(self, X, Y):
        """ Train this tree echo state network on the trees X with targets Y.
        This uses classic least squares for training and is thus quite fast.

        Args:
        X:              A list of m trees in node list/adjacency list format.
        Y:              A m x dim_out matrix of targets for each data point.
        """
        # encode all trees in parallel
        H = self.encode_list(X)
        # then perform the pseudo-inverse computation
        inv = np.linalg.inv(np.dot(H.T, H) + self._regul * np.eye(self._dim))
        self._W_out = np.dot(np.dot(Y.T, H), inv)

        # compute the prediction
        Y_pred = np.dot(H, self._W_out.T)

        # and compute the bias
        self._b_out = np.mean(Y - Y_pred)

    def predict(self, X):
        """ Predicts an output matrix for a given list of trees.

        Args:
        X: A list of m trees in node list/adjacency list format.

        Returns: A m x dim_out matrix of predictions.
        """
        # encode all trees in parallel
        H = self.encode_list(X)
        # then construct the output
        return np.dot(H, self._W_out.T) + self._b_out

class TreeEchoStateNetClassifier(TreeEchoStateNet):
    """An extension of tree echo state networks to perform classification.

    This class produces a K dimensional output for K classes and then assigns
    the class with maximum activation.

    Attributes:
    _dim:       The encoding dimensionality.
    _labels:    The list of unique classes in the training data.
    _alphabet:  A dictionary mapping alphabet symbols to indices.
    _Ws:        A K x _dim x _dim tensor containing the weight matrices
                for each symbol, where K is the alphabet size.
    _bs:        A K x _dim matrix containing the bias vectors
                for each symbol, where K is the alphabet size.
    _W_out:     The output weight matrix.
    _b_out:     The output bias.
    _regul:     The L2 regularization strength.
    _n_jobs:    The number of jobs for parallel encoding.
    """
    def __init__(self, dim, alphabet, sparsity = 0.1, radius = 0.9, regul = 1E-5, n_jobs = 8):
        """ Initializes a tree echo state network.

        Args:
        dim:      The encoding dimensionality
        alphabet: A list or set of unique symbols which may occur in input
                  trees.
        sparsity: The proportion of nonzero entries in the weight matrices.
                  Defaults to 0.1.
        radius:   The scaling factor for the weight matrices to ensure a
                  contractive mapping.
        regul:     The L2 regularization strength. Defaults to 1E-5.
        n_jobs:    The number of jobs for parallel encoding. Defaults to 8.
        """
        super(TreeEchoStateNetClassifier, self).__init__(dim, alphabet, sparsity, radius, regul, n_jobs)

    def predict(self, X):
        """ Classifies every tree in a given list of trees.

        Args:
        X: A list of trees in node list/adjacency list format.

        Returns: The predicted class labels as a len(X) vector.
        """
        # compute the scores for all input trees
        Y_scores = super(TreeEchoStateNetClassifier, self).predict(X)
        # retrieve the maximum score index
        Y_pred = np.argmax(Y_scores, axis=1)
        # map it to the correct labels
        Y_pred = self._labels[Y_pred]
        return Y_pred

    def fit(self, X, Y, minibatch_size = 5, step_size = 1E-3, max_iterations = None, threshold = 1E-2, print_step = None):
        """ Train this tree echos tate netowrk classifier on the trees X with
        labels Y.

        The training is done by translating the labels to a one-hot coding and
        then performing regression via the parent class.

        Args:
        X:              A list of m trees in node list/adjacency list format.
        Y:              A list of class labels for the trees.
        """
        # identify unique labels
        self._labels = np.unique(Y)
        L = len(self._labels)
        # construct a mapping from labels to unique label indices
        label_map = {}
        for l in range(L):
            label_map[self._labels[l]] = l
        # construct a one-hot coding of the input
        Y_one_hot = np.zeros((len(X), L))
        for i in range(len(X)):
            Y_one_hot[i, label_map[Y[i]]] = 1.
        # train the network via regression
        super(TreeEchoStateNetClassifier, self).fit(X, Y_one_hot)
