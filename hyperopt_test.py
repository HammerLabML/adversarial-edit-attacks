#!/usr/bin/python3
"""
Tests hyperparameter optimization.

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
import hyperopt

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

class TestHyperopt(unittest.TestCase):

    def test_grid_search_cv_svm(self):
        # consider three kernels, to of which are bad and one of which is good
        # for the given labels
        Y = np.array([-1, -1, 1, 1])
        Ks = [
            -np.outer(Y, Y), # bad kernel
            np.zeros((len(Y), len(Y))), # bad kernel
            np.outer(Y, Y) # good kernel
        ]
        # perform hyper parameter optimization
        C, K, accs = hyperopt.grid_search_cv_svm(Y, [1.], Ks, n_splits = 2)
        # check results
        self.assertEqual(1., C)
        self.assertEqual(2, K)
        accs_expected = [[[0., 0.5, 1.]], [[0., 0.5, 1.]]]
        np.testing.assert_array_equal(accs_expected, accs)

    def test_grid_search_cv_echo_state(self):
        # use a data set which is easy but which becomes hard for ridiculous
        # hyper-parameter settings

        X = []
        X.append((['A', 'B'], [[1], []]))
        X.append((['A', 'A', 'B'], [[1, 2], [], []]))
        X.append((['A', 'B', 'B'], [[1, 2], [], []]))
        X.append((['A', 'B', 'B'], [[1], [2], []]))
        X.append((['A', 'A', 'A', 'B'], [[1], [2, 3], [], []]))
        X.append((['A', 'B', 'A', 'A'], [[1], [2], [3], []]))
        Y = np.array([-1, +1, -1, -1, +1, +1])

        # this is unsolvable for a single neuron or for a trivial spectral
        # radius
        dims = [1, 10]
        radii = [1E-3, 0.9]

        dim, radius, accs = hyperopt.grid_search_cv_echo_state(X, Y, ['A', 'B'], dims, radii, n_splits = 2)
        # check the results
        self.assertEqual(10, dim)
        self.assertEqual(0.9, radius)
        # we do not check the accuracies because the training is not always 100% reliable.

if __name__ == '__main__':
    unittest.main()
