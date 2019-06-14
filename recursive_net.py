"""
Implements recursive neural networks in pytorch.

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

import numpy as np
import torch
from tree_utils import root

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

class RecursiveNet(torch.nn.Module):
    """ A recursive neural net as a pyTorch module.

    This particular recursive neural network module works by learning
    for each symbol a single neural network layer that is applied to the
    sum of the encodings of its children in the tree. This way, we avoid
    having to specify the exact number of children for each possible symbol.

    Attributes:
    _dim:       The encoding dimensionality.
    _out_dim:   The output dimensionality
    _layers:    A module dictionary which maps each symbol to a layer.
    _nonlin:    A nonlinearity for the layer outputs.
    _out_layer: A single layer that maps from the encoding of the tree to an
                output vector.
    """
    def __init__(self, dim, out_dim, alphabet, nonlin = torch.nn.Sigmoid()):
        super(RecursiveNet, self).__init__()
        self._dim = dim
        self._out_dim = out_dim
        self._layers = torch.nn.ModuleDict()
        for symbol in alphabet:
            self._layers[symbol] = torch.nn.Linear(self._dim, self._dim)
        self._nonlin = nonlin
        self._out_layer = torch.nn.Linear(self._dim, self._out_dim)

    def forward(self, x_nodes, x_adj):
        """ Computes the output value for a single tree in node list/
        adjacency list format.

        Args:
        x_nodes: The node list of the input tree.
        x_adj:   The adjacency list of the input tree.

        Returns: a _out_dim dimensional vector.
        """
        # first, identify the root of the tree
        r = root(x_adj)
        # initialize an encoding matrix for all nodes
        encodings = torch.zeros([len(x_nodes), self._dim])
        has_encoding = np.zeros(len(x_nodes), dtype=bool)
        # then, we use a stack to move recursively through the tree and
        # embed all nodes
        stk = [r]
        while(stk):
            # pop the most recent node from the stack
            i = stk.pop()
            sym_i = x_nodes[i]
            # handle the special case of leaves
            if(not x_adj[i]):
                # if i is a leaf, we can embed it using the layer with zero
                # input
                encodings[i, :] = self._nonlin(self._layers[sym_i](torch.zeros(self._dim)))
                has_encoding[i] = True
            elif(np.all(has_encoding[x_adj[i]])):
                # if this is not a leaf but all children are already encoded,
                # we can add the encodings of all children and then put that
                # into the symbol layer
                children_encoding = torch.mean(encodings[x_adj[i], :], 0)
                encodings[i, :] = self._nonlin(self._layers[sym_i](children_encoding))
                has_encoding[i] = True
            else:
                # if the children are not yet embedded, we first push i back
                # on the stack and then all its children
                stk.append(i)
                for j in x_adj[i]:
                    stk.append(j)
        # after this while loop is completed, all nodes are embedded and we
        # can return the output for the tree as the encoding of the root,
        # pushed through the output layer
        return self._out_layer(encodings[r, :])

class RecursiveNetClassifier(RecursiveNet):
    """An extension of recursive neural networks to perform classification.

    This class produces a K dimensional output for K classes and then assigns
    the class with maximum activation.

    Attributes:
    _dim:       The encoding dimensionality.
    _out_dim:   The number of classes.
    _layers:    A module dictionary which maps each symbol to a layer.
    _nonlin:    A nonlinearity for the layer outputs.
    _out_layer: A single layer that maps from the encoding of the tree to an
                output vector.
    _labels:    A list of possible labels.
    """
    def __init__(self, dim, unique_labels, alphabet):
        super(RecursiveNetClassifier, self).__init__(dim, len(unique_labels), alphabet, torch.nn.Tanh())
        self._labels  = unique_labels

    def predict(self, X):
        """ Classifies every tree in a given list of trees.

        Args:
        X: A list of trees in node list/adjacency list format.

        Returns: The predicted class labels as a list
        """
        Y_pred = []
        for i in range(len(X)):
            # compute the class activations
            acts = self.forward(X[i][0], X[i][1])
            # get the argmax
            label_index = torch.argmax(acts).item()
            # the label corresponding to that index is our prediction
            Y_pred.append(self._labels[label_index])
        return Y_pred

    def fit(self, X, Y, minibatch_size = 5, step_size = 1E-3, max_iterations = None, threshold = 1E-2, print_step = None):
        """ Train this recursive net on the trees X with labels Y.

        This uses a log loss ADAM as an optimizer with the given optimizer
        parameters.

        Args:
        X:              A list of m trees in node list/adjacency list format.
        Y:              A list of class labels for the trees.
        minibatch_size: The minibatch size for training. Defaults to 5.
        step_size:      The learning rate or step size for training. Defaults
                        to 1E-3.
        max_iterations: A maximum number of iterations. Defaults to infinite.
        threshold:      A threshold on the error, i.e. if the error is below
                        threshold, we stop. Defaults to 1E-2
        print_step:     Every print_step iterations, we print the currently
                        lowest loss.
        """
        # set up the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=step_size)
        # set up the loss
        loss_fun  = torch.nn.CrossEntropyLoss()
        # start iterating
        learning_curve = []
        while((len(learning_curve) < 10 or learning_curve[-1] > threshold) and
            ((max_iterations is None or len(learning_curve) < max_iterations))):
            # zero the gradient
            optimizer.zero_grad()
            # draw minibatch_size random trees from the training data
            minibatch = np.random.randint(len(X), size=minibatch_size)
            # perform the predictions for the minibatch
            # and get the label index for each point
            batch_predictions = []
            y_actual = []
            for i in minibatch:
                batch_predictions.append(self.forward(X[i][0], X[i][1]))
                y_actual.append(self._labels.index(Y[i]))
            # stack the predictions to a torch tensor
            y_pred   = torch.stack(batch_predictions)
            y_actual = torch.tensor(y_actual, dtype=torch.long)
            # compute the loss
            loss_obj = loss_fun(y_pred, y_actual)
            # compute the gradient
            loss_obj.backward()
            # store the loss
            learning_curve.append(loss_obj.item())
            # perform an optimizer step
            optimizer.step()
            # print the loss if so desired
            if(print_step is not None and len(learning_curve) % print_step == 0):
                min_loss = np.min(learning_curve)
                print('lowest loss after %d steps: %g' % (len(learning_curve), min_loss))
        # return the learning curve
        return learning_curve
