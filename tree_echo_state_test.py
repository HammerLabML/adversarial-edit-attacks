#!/usr/bin/python3
"""
Tests tree echo state networks.

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
import tree_echo_state

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

class TestTreeEchoStateNet(unittest.TestCase):

    def test_echo_state_net(self):
        # consider the example of counting the number of 'A's in a tree.
        X = []
        X.append((['A', 'B'], [[1], []]))
        X.append((['A', 'A', 'B'], [[1, 2], [], []]))
        X.append((['A', 'B', 'B'], [[1, 2], [], []]))
        X.append((['A', 'B', 'B'], [[1], [2], []]))
        X.append((['A', 'A', 'A', 'B'], [[1], [2, 3], [], []]))
        X.append((['A', 'B', 'A', 'A'], [[1], [2], [3], []]))
        Y = np.array([1, 2, 1, 1, 3, 3])
        Y = np.expand_dims(Y, 1)

        # train a tree echo state net on this task
        rec_net = tree_echo_state.TreeEchoStateNet(10, ['A', 'B'])
        rec_net.fit(X, Y)

        # check the predictions
        Y_pred = rec_net.predict(X)
        self.assertTrue(np.mean(np.abs(Y_pred - Y)) < 0.1)

    def test_tree_echo_state_net_classifier(self):
        # consider the example of counting the number of 'A's in a tree.
        # the true label is +1 if the number of 'A's is bigger than the number
        # of 'B's and -1 otherwise.
        X = []
        X.append((['A', 'B'], [[1], []]))
        X.append((['A', 'A', 'B'], [[1, 2], [], []]))
        X.append((['A', 'B', 'B'], [[1, 2], [], []]))
        X.append((['A', 'B', 'B'], [[1], [2], []]))
        X.append((['A', 'A', 'A', 'B'], [[1], [2, 3], [], []]))
        X.append((['A', 'B', 'A', 'A'], [[1], [2], [3], []]))
        Y = np.array([-1, +1, -1, -1, +1, +1])

        # train a tree echo state net on this task
        rec_net = tree_echo_state.TreeEchoStateNetClassifier(10, ['A', 'B'])
        rec_net.fit(X, Y)

        # check the predictions
        Y_pred = rec_net.predict(X)
        np.testing.assert_array_equal(Y, Y_pred)

if __name__ == '__main__':
    unittest.main()
