#!/usr/bin/python3
"""
Tests the pytorch recursive neural net implementation.

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
import recursive_net

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

class TestRecursiveNet(unittest.TestCase):

    def test_recursive_net(self):
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

        # train a recursive net on this task
        rec_net = recursive_net.RecursiveNetClassifier(3, [-1, +1], ['A', 'B'])
        learning_curve = rec_net.fit(X, Y, print_step = 100)

        # we should achieve a very low loss on this one
        self.assertTrue(np.min(learning_curve) < 0.1)
        # and the predictions should be correct for all points
        Y_pred = rec_net.predict(X)
        np.testing.assert_array_equal(Y, Y_pred)

if __name__ == '__main__':
    unittest.main()
