"""
Implements adversarial edits for tree structured data, in particular the random
edit strategy in the construct_random_adversarial and
construct_random_adversarials methods, and the backatracing strategy in the
construct_adversarial and construct_adversarials methods.

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

import copy
import random
import numpy as np
import ted
import multiprocess as mp
import tree_edits
import tree_utils

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

def construct_adversarials(X, D, Y, Y_pred, classifier):
    """ Constructs an adversarial example for each correctly classified data
    point, based on the given labels.

    In more detail, this methods infers for each correctly classified data
    point i the closest correctly classified data point j with a different
    label and then calls construct_adversarial(X[i], X[j], Y[j], classifier)
    to construct an adversarial.

    Args:
    X:          A list of trees in adjacency list/node list format.
    D:          The pairwise edit distance matrix between all X.
    Y:          The actual labels for these trees.
    Y_pred:     (optional) The labels for these trees as predicted by the classifier.
    classifier: A function that maps a tree (in node list/adjacency format) to
                a label.

    Returns:
    Z:          A list of adversarial examples in adjacency list/node list
                format with the same length as X. For misclassified data
                points, the list contains a None entry.
    labels:     The new labels for the adversarial examples. If Z[i] is None,
                we have labels[i] = not a number.
    ds:         The relative distances from the points to their adversarial
                example, i.e. for each i: ds[i] = d(X[i], Z[i])/d(Z[i], X[j]),
                where j is the closest correctly classified data point to Z[i]
                with the same label. If Z[i] = None, we have D[i] = inf.
    """
    if(len(X) != len(Y)):
        raise ValueError('Expected one label per input tree')
    if(D.shape[0] != len(X)):
        raise ValueError('Expected one distance matrix row per input tree')
    if(D.shape[1] != len(X)):
        raise ValueError('Expected one distance matrix column per input tree')
    # convert labels to numpy arrays
    Y = np.array(Y)
    if(Y_pred is not None):
        if(len(Y_pred) != len(X)):
            raise ValueError('Expected one predicted label per input tree')
        Y_pred = np.array(Y_pred)
    else:
        Y_pred = np.zeros(len(X))
        for i in range(len(X)):
            x_nodes, x_adj = X[i]
            Y_pred[i] = classifier(x_nodes, x_adj)
    # initialize outputs
    Z  = [None] * len(X)
    labels = np.full(len(X), np.nan)
    ds = np.full(len(X), np.inf)

    # get correctly classified data points
    correct_index = np.where(Y == Y_pred)[0]

    # iterate over all correctly classified data points
    for i in correct_index:
        # find the closest point to x which is correctly classified but with a
        # different label
        dmin = np.inf
        jmin = -1
        for j in correct_index:
            if(Y[i] != Y[j] and D[i,j] < dmin):
                dmin = D[i,j]
                jmin = j
        if(jmin < 0):
            # if we did not find such a data point, we have to break off
            # our attempt, unfortunately
            continue
        # retrieve the start and end point for our adversarial construction
        x_nodes, x_adj = X[i]
        x_label = Y[i]
        y_nodes, y_adj = X[jmin]

        # construct an adversarial example leading from x to y
        z_nodes, z_adj, script, labels[i] = construct_adversarial(
                                        x_nodes, x_adj, x_label,
                                        y_nodes, y_adj, classifier)
        # check the new label and use it to compute the relative
        # distance d(X[i], Z[i])/d(Z[i], X[j])
        if(labels[i] == Y[jmin]):
            d_zy = D[i, jmin] - len(script)
        else:
            # compute the distance to all data points with the same label
            X_same_class = []
            for j in np.where(Y == labels[i])[0]:
                X_same_class.append(X[j])
            ds_same_class = mp.pairwise_distances([(z_nodes, z_adj)], X_same_class)[0]
            d_zy = np.min(ds_same_class)
        # store the relative distance
        if(d_zy > 0):
            ds[i] = float(len(script)) / float(d_zy)

        #store the adversarial example
        Z[i] = (z_nodes, z_adj)

    # return the results
    return Z, labels, ds

def construct_adversarial(x_nodes, x_adj, x_label, y_nodes, y_adj, classifier):
    """ Finds the shortest possible edit script which reduces the distance
    between the trees x and y, such that the predicted label of the resulting
    tree is different from x_label.

    Note that this relies on the standard tree edit distance.

    Args:
    x_nodes: the node list of tree x.
    x_adj:   the adjacency list of tree x.
    x_label: a numeric label for tree x.
    y_nodes: the node list of tree y.
    y_adj:   the adjacency list of tree y.
    classifier: a function that maps a tree (in node list/adjacency format) to
                a label.

    Returns:
    z_nodes: the node list of the adversarial example.
    z_adj:   the adjacency list of the adversarial example.
    script:  the edit script such that script.apply(x_nodes, x_adj) =
             (z_nodes, z_adj)
    label:   The new label for the adversarial example.
    """
    # construct the shortest edit script from x to y.
    trace   = ted.standard_ted_backtrace(x_nodes, x_adj, y_nodes, y_adj)
    script  = tree_edits.trace_to_script(trace, x_nodes, x_adj, y_nodes, y_adj)
    # perform a binary search to identify the shortest edit script which still
    # flips the label
    return _binary_search(x_nodes, x_adj, x_label, script, classifier)


def _binary_search(x_nodes, x_adj, x_label, script, classifier, lo = 0, hi = None):
    """ Applies as little edits of the script to the given tree as possible
        such that the label of the given tree still changes.

    Args:
    x_nodes: The node list of the input tree.
    x_adj:   The adjacency list of the input tree.
    x_label: The label of the input tree.
    script:  An edit script such that the resulting script has a label other
             than x_label, i.e. classifier(script.apply(x_nodes, x_adj)) !=
             x_label.
    classifier: a function that maps a tree (in node list/adjacency format) to
                a label.
    lo: (optional) The minimum amount of edits necessary (defaults to zero).
    hi: (optional) The maximum amount of edits necessary (defaults to
                   len(script)).

    Returns:
    z_nodes: The node list of the adversarial example.
    z_adj:   The adjacency list of the adversarial example.
    script:  The edit script such that script.apply(x_nodes, x_adj) =
             (z_nodes, z_adj)
    label:   The new label for the adversarial example.
    """
    if(hi is None):
        hi = len(script)
    # continue the binary search until our lower bound reaches the upper bound
    while(lo < hi):
        # update the median
        d = int(0.5 * (lo + hi))
        # apply the script until d
        z_nodes, z_adj = tree_edits.Script(script[:d]).apply(x_nodes, x_adj)
        # classify our current tree
        label = classifier(z_nodes, z_adj)
        # check whether the label has changed
        if(label == x_label):
            # if the label has not yet changed, increase the lower bound
            lo = d + 1
        else:
            # otherwise, decrease the upper bound
            hi = d
    # after performing the binary sarch, lo == hi and we apply the script
    # one last time
    script = tree_edits.Script(script[:hi])
    z_nodes, z_adj = script.apply(x_nodes, x_adj)
    label = classifier(z_nodes, z_adj)
    # verify the desired result
    if(label == x_label):
        raise ValueError('Internal error: Binary search did not achieve a label change.')

    return z_nodes, z_adj, script, label


def construct_random_adversarials(X, Y, Y_pred, classifier, alphabet = None, max_d = None):
    """ Constructs a random adversarial example for each correctly classified data
    point, based on the given labels.

    In more detail, this methods applied random edits to each correctly
    classified data point until the label flips to something else.

    Args:
    X:          A list of trees in adjacency list/node list format.
    Y:          The actual labels for these trees.
    Y_pred:     The labels for these trees as predicted by the classifier.
    classifier: A function that maps a tree (in node list/adjacency format) to
                a label.
    alphabet:   (optional) a list of possible symbols; if not given, this is
                inferred from the data.
    max_d:      (optional) an upper bound on the number of random edits to
                apply.

    Returns:
    Z:          A list of adversarial examples in adjacency list/node list
                format with the same length as X. For misclassified data
                points or data points where the maximum number of edits was
                crossed, this list contains a None entry.
    labels:     The new labels for the adversarial examples.
    ds:         The relative distances from the points to their adversarial
                example, i.e. for each i: ds[i] = d(X[i], Z[i])/d(Z[i], X[j]),
                where j is the closest correctly classified data point to Z[i]
                with the same label as Z[i]. If Z[i] = None, we have D[i] = inf.
    """
    if(alphabet is None):
        alphabet = set()
        for x_nodes, x_adj in X:
            alphabet.update(x_nodes)
        alphabet = list(alphabet)
        print(alphabet)
    # convert labels to numpy arrays
    Y = np.array(Y)
    Y_pred = np.array(Y_pred)
    # initialize outputs
    Z  = [None] * len(X)
    labels = np.zeros(len(X))
    ds = np.full(len(X), np.inf)

    # get correctly classified data points
    correct_index = np.nonzero(Y == Y_pred)[0]

    # iterate over all correctly classified data points
    for i in correct_index:
        # construct an adversarial example for the ith data point
        z_nodes, z_adj, script, labels[i] = construct_random_adversarial(
                                        X[i][0], X[i][1], Y[i],
                                        classifier, alphabet, max_d)
        if(z_nodes is None):
            continue
        #store it
        Z[i] = (z_nodes, z_adj)
        # compute the distance to all data points with the same label
        X_same_class = []
        for j in np.where(Y == labels[i])[0]:
            X_same_class.append(X[j])
        d_zy = np.min(mp.pairwise_distances([(z_nodes, z_adj)], X_same_class)[0])
        if(d_zy > 0):
            ds[i] = float(len(script)) / float(d_zy)

    # return the results
    return Z, labels, ds

def construct_random_adversarial(x_nodes, x_adj, x_label, classifier, alphabet, max_d = None):
    """ Applies random edits to the tree x until the label flips to something
    different.

    Args:
    x_nodes:    the node list of tree x.
    x_adj:      the adjacency list of tree x.
    x_label:    the predicted label of the classifier for x.
    classifier: a function that maps a tree (in node list/adjacency format) to
                a label.
    alphabet:   the alphabet of possible node labels.
    max_d:      a maximum number of edits. If this is crossed, the method stops
                and returns none.

    Returns:
    z_nodes: the node list of the adversarial example.
    z_adj:   the adjacency list of the adversarial example.
    script:  the edit script such that script.apply(x_nodes, x_adj) =
             (z_nodes, z_adj)
    label:   the new label for the adversarial example.
    """
    # initialize script
    script = tree_edits.Script()
    # get the current label of the input tree
    y_pred = x_label
    # perform a copy of the input tree to which we apply our edits
    z_nodes = copy.copy(x_nodes)
    z_adj = copy.deepcopy(x_adj)
    # a minimum and maximum estimate of edits we need to flip the label
    lo = 0
    hi = 0
    # perform binary search, i.e. double the amount of edits until the label flips
    while(y_pred == x_label and (max_d is None or len(script) < max_d)):
        # update lo and hi
        lo = len(script)
        hi = 2 * len(script) + 1
        # generate len(script)+1 new edits
        for t in range(len(script) + 1):
            # select randomly whether we delete, replace, or insert
            edit_type = random.randrange(3)
            if(edit_type == 0):
                # choose a random node to delete (except the root node)
                if(len(z_nodes) < 2):
                    edit_type = random.randrange(2) + 1
                else:
                    i = random.randrange(len(z_nodes) - 1) + 1
                    edit = tree_edits.Deletion(i)
            if(edit_type == 1):
                # choose a random node to replace
                i = random.randrange(len(z_nodes))
                # and choose a random target label
                label = alphabet[random.randrange(len(alphabet))]
                edit = tree_edits.Replacement(i, label)
            else:
                # choose a random parent for insertion
                p = random.randrange(len(z_nodes))
                # choose a random child index
                c = random.randrange(len(z_adj[p]) + 1)
                # choose a random target label to insert
                label = alphabet[random.randrange(len(alphabet))]
                # and choose a random number of children
                if(len(z_adj[p]) > c):
                    C = random.randrange(len(z_adj[p]) - c)
                else:
                    C = 0
                edit = tree_edits.Insertion(p, c, label, C)
            # apply the edit
            edit.apply_in_place(z_nodes, z_adj)
            # and add it to the script
            script.append(edit)
        # update the prediction
        y_pred = classifier(z_nodes, z_adj)
    if(max_d is not None and len(script) >= max_d):
        return None, None, None, None
    # once the label is flipped, perform a binary search to find how many
    # edit we really need
    return _binary_search(x_nodes, x_adj, x_label, script, classifier, lo)
