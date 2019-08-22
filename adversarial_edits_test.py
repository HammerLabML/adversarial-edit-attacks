#!/usr/bin/python3
"""
Tests adversarial edit construction

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
import unittest
import numpy as np
import edist.tree_edits as tree_edits
import edist.tree_utils as tree_utils
import adversarial_edits

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

class TestAdversarialEdits(unittest.TestCase):

    def test_construct_adversarial(self):
        # consider the example of counting the number of 'A's in a tree.
        # the true label is +1 if the number of 'A's is bigger than the number
        # of 'B's and -1 otherwise. However, we have a classifier
        # that always returns +1 if the number of 'B's is smaller than 2,
        # which works on the training data. In particular, consider the
        # following two trees
        x_nodes = ['A', 'A', 'B', 'A', 'A']
        x_adj   = [[1], [2], [3], [4], []]
        y_nodes = ['A', 'B', 'B']
        y_adj   = [[1], [2], []]
        y_label = -1
        # and the following classifier
        def classifier(x_nodes, x_adj):
            if(x_nodes.count('B') < 2):
                return +1
            else:
                return -1
        # then, the adversarial example should replace one A with a B,
        # resulting in a tree like A(B(B(A(A)))), which would have the
        # 'true' label +1, but has the predicted label -1.

        # Note that our construction has no clue about the 'true' label;
        # it just works based on edit scripts between training data points
        z_nodes, z_adj, script, label = adversarial_edits.construct_adversarial(
                                        x_nodes, x_adj, +1, y_nodes, y_adj,
                                        classifier)
        self.assertEqual(-1, classifier(z_nodes, z_adj))
        self.assertEqual(2, z_nodes.count('B'))
        self.assertEqual(1, len(script))
        self.assertEqual(-1, label)

    def test_construct_adversarials(self):
        # consider the same example as above
        x_nodes = ['A', 'A', 'B', 'A', 'A']
        x_adj   = [[1], [2], [3], [4], []]
        y_nodes = ['A', 'B', 'B']
        y_adj   = [[1], [2], []]

        X = [(x_nodes, x_adj), (y_nodes, y_adj)]
        D = np.array([[0, 3], [3, 0]])
        Y = [+1, -1]

        # with the following classifier
        def classifier(x_nodes, x_adj):
            if(x_nodes.count('B') < 2):
                return +1
            else:
                return -1
        Z, labels, ds = adversarial_edits.construct_adversarials(X, D, Y, Y, classifier)
        self.assertEqual(-1, labels[0])
        self.assertEqual(0.5, ds[0])
        self.assertEqual(+1, labels[1])
        self.assertEqual(0.5, ds[1])

    def test_construct_random_adversarial(self):
        # consider the example of counting the number of 'A's in a tree.
        # the true label is +1 if the number of 'A's is bigger than the number
        # of 'B's and -1 otherwise. However, we have a classifier
        # that always returns +1 if the number of 'B's is smaller than 2,
        # which works on the training data. In particular, consider the
        # following tree
        x_nodes = ['A', 'A', 'B', 'A', 'A']
        x_adj   = [[1], [2], [3], [4], []]
        x_label = 1
        # and the following classifier
        def classifier(x_nodes, x_adj):
            if(x_nodes.count('B') < 2):
                return +1
            else:
                return -1
        # then, the adversarial example should add a B at some point

        # Note that our construction has no clue about the 'true' label;
        # it just works based on edit scripts between training data points
        z_nodes, z_adj, script, label = adversarial_edits.construct_random_adversarial(
                                        x_nodes, x_adj, x_label, classifier, ['A', 'B'])
        self.assertEqual(-1, label)
        self.assertEqual(2, z_nodes.count('B'))
        self.assertTrue(len(script) >= 1)

    def test_construct_random_adversarials(self):
        # perform the same test as above
        x_nodes = ['A', 'A', 'B', 'A', 'A']
        x_adj   = [[1], [2], [3], [4], []]
        y_nodes = ['A', 'B', 'B']
        y_adj   = [[1], [2], []]


        X = [(x_nodes, x_adj), (y_nodes, y_adj)]
        Y = [+1, -1]

        # and the following classifier
        def classifier(x_nodes, x_adj):
            if(x_nodes.count('B') < 2):
                return +1
            else:
                return -1

        Z, labels, ds = adversarial_edits.construct_random_adversarials(X, Y, [+1,+1], classifier, ['A', 'B'])
        np.testing.assert_array_equal([-1, 0], labels)
        self.assertEqual(2, len(Z))
        self.assertTrue(Z[0][0].count('B') >= 2)
        self.assertTrue(Z[1] is None)
        self.assertEqual(2, len(ds))

    def test_binary_search(self):
        # construct a trivial start tree
        x_nodes = ['A']
        x_adj   = [[]]
        # construct a trivial script where we only add As
        n = 10
        script  = []
        for i in range(n-1):
            script.append(tree_edits.Insertion(i, 0, 'A'))
        # use a classifier which switches the label to 1 if we have
        # more than m As
        m = n-1
        def classifier(nodes, adj):
            if(nodes.count('A') > m):
                return +1
            else:
                return -1
        # accordingly, our expected script adds precisely m As
        expected_script = tree_edits.Script(script[:m])
        z_nodes_expected, z_adj_expected = expected_script.apply(x_nodes, x_adj)
        self.assertEqual(-1, classifier(x_nodes, x_adj))
        self.assertEqual(+1, classifier(z_nodes_expected, z_adj_expected))
        # perform the binary search
        z_nodes_actual, z_adj_actual, actual_script, label = adversarial_edits._binary_search(x_nodes, x_adj, -1, script, classifier)
        self.assertEqual(z_nodes_expected, z_nodes_actual)
        self.assertEqual(z_adj_expected, z_adj_actual)
        self.assertEqual(expected_script, actual_script)
        self.assertEqual(+1, label)

if __name__ == '__main__':
    unittest.main()
