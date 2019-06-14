#!/usr/bin/python3
"""
Tests the tree edit distance implementation.

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
import time
import numpy as np
import tree_utils
import tree_edits
from trace import Trace
import ted

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

class TestTED(unittest.TestCase):

    def test_outermost_right_leaves(self):
        # consider the empty tree
        np.testing.assert_array_equal(np.array([], dtype=int), ted.outermost_right_leaves([]))
        # consider the tree 0(1(2), 3(4, 5))
        adj = [[1, 3], [2], [], [4, 5], [], []]
        expected_orl = np.array([5, 2, 2, 5, 4, 5], dtype=int)
        actual_orl = ted.outermost_right_leaves(adj)
        np.testing.assert_array_equal(expected_orl, actual_orl)

    def test_keyroots(self):
        # consider the empty tree
        np.testing.assert_array_equal(np.array([], dtype=int), ted.keyroots(np.array([], dtype=int)))
        # consider the tree 0(1(2), 3(4, 5))
        orl = np.array([5, 2, 2, 5, 4, 5], dtype=int)
        expected_keyroots = np.array([4, 1, 0], dtype=int)
        actual_keyroots = ted.keyroots(orl)
        np.testing.assert_array_equal(expected_keyroots, actual_keyroots)

    def test_ted(self):
        # consider three example trees, one of them being empty
        x = []
        x_adj = []
        # the tree a(b(c, d), e)
        y = ['a', 'b', 'c', 'd', 'e']
        y_adj = [[1, 4], [2, 3], [], [], []]
        # the tree f(g)
        z = ['f', 'g']
        z_adj = [[1], []]

        trees = [x, y, z]
        adjs  = [x_adj, y_adj, z_adj]

        m = len(trees)

        # as character distance, use the kronecker delta
        def kron_distance(x, y):
            if(x == y):
                return 0.
            else:
                return 1.

        # set up the expected distances
        D_expected = np.array([[0., 5., 2.], [5., 0., 5.], [2., 5., 0.]])

        # compute the actual matrix of pairwise distances
        D_actual = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                D_actual[i, j] = ted.ted(trees[i], adjs[i], trees[j], adjs[j], kron_distance)

        np.testing.assert_array_equal(D_expected, D_actual)

    def test_ted_backtrace(self):
        # consider two example trees
        # the tree a(b(c, d), e)
        y = ['a', 'b', 'c', 'd', 'e']
        y_adj = [[1, 4], [2, 3], [], [], []]
        # and the tree f(g)
        z = ['f', 'g']
        z_adj = [[1], []]

        # as character distance, use the kronecker delta
        def kron_distance(x, y):
            if(x == y):
                return 0.
            else:
                return 1.

        # set up expected trace
        expected_trace = Trace()
        expected_trace.append_operation(0, -1)
        expected_trace.append_operation(1, 0)
        expected_trace.append_operation(2, -1)
        expected_trace.append_operation(3, 1)
        expected_trace.append_operation(4, -1)

        # compute actual trace
        actual_trace = ted.ted_backtrace(y, y_adj, z, z_adj, kron_distance)
        # check result
        self.assertEqual(expected_trace, actual_trace)

        # test a case where the mapping may be mislead
        x = ['block', 'for', 'block', 'expression', 'method', 'return', 'method', 'member', 'identifier']
        x_adj = [[1, 5], [2], [3], [4], [], [6], [7], [8], []]
        y = ['block', 'for', 'block', 'expression', 'method', 'block', 'return', 'literal', 'return']
        y_adj = [[1, 8], [2], [3, 5], [4], [], [6], [7], [], []]

        # set up expected trace
        expected_trace = Trace()
        expected_trace.append_operation(0, 0)
        expected_trace.append_operation(1, 1)
        expected_trace.append_operation(2, 2)
        expected_trace.append_operation(3, 3)
        expected_trace.append_operation(4, 4)
        expected_trace.append_operation(-1, 5)
        expected_trace.append_operation(-1, 6)
        expected_trace.append_operation(-1, 7)
        expected_trace.append_operation(5, 8)
        expected_trace.append_operation(6, -1)
        expected_trace.append_operation(7, -1)
        expected_trace.append_operation(8, -1)

        # compute actual trace
        actual_trace = ted.ted_backtrace(x, x_adj, y, y_adj, kron_distance)
        # check result
        self.assertEqual(expected_trace, actual_trace)

    def test_standard_ted(self):
        # consider three example trees, one of them being empty
        x = []
        x_adj = []
        # the tree a(b(c, d), e)
        y = ['a', 'b', 'c', 'd', 'e']
        y_adj = [[1, 4], [2, 3], [], [], []]
        # the tree f(g)
        z = ['f', 'g']
        z_adj = [[1], []]

        trees = [x, y, z]
        adjs  = [x_adj, y_adj, z_adj]

        m = len(trees)

        # set up the expected distances
        D_expected = np.array([[0, 5, 2], [5, 0, 5], [2, 5, 0]], dtype=int)

        # compute the actual matrix of pairwise distances
        D_actual = np.zeros((m, m), dtype=int)
        for i in range(m):
            for j in range(m):
                D_actual[i, j] = ted.standard_ted(trees[i], adjs[i], trees[j], adjs[j])

        np.testing.assert_array_equal(D_expected, D_actual)

    def test_standard_ted_backtrace(self):
        # consider two example trees
        # the tree a(b(c, d), e)
        y = ['a', 'b', 'c', 'd', 'e']
        y_adj = [[1, 4], [2, 3], [], [], []]
        # and the tree f(g)
        z = ['f', 'g']
        z_adj = [[1], []]

        # set up expected trace
        expected_trace = Trace()
        expected_trace.append_operation(0, -1)
        expected_trace.append_operation(1, 0)
        expected_trace.append_operation(2, -1)
        expected_trace.append_operation(3, 1)
        expected_trace.append_operation(4, -1)

        # compute actual trace
        actual_trace = ted.standard_ted_backtrace(y, y_adj, z, z_adj)
        # check result
        self.assertEqual(expected_trace, actual_trace)

        # test a case where the mapping may be mislead
        x = ['block', 'for', 'block', 'expression', 'method', 'return', 'method', 'member', 'identifier']
        x_adj = [[1, 5], [2], [3], [4], [], [6], [7], [8], []]
        y = ['block', 'for', 'block', 'expression', 'method', 'block', 'return', 'literal', 'return']
        y_adj = [[1, 8], [2], [3, 5], [4], [], [6], [7], [], []]

        # set up expected trace
        expected_trace = Trace()
        expected_trace.append_operation(0, 0)
        expected_trace.append_operation(1, 1)
        expected_trace.append_operation(2, 2)
        expected_trace.append_operation(3, 3)
        expected_trace.append_operation(4, 4)
        expected_trace.append_operation(-1, 5)
        expected_trace.append_operation(-1, 6)
        expected_trace.append_operation(-1, 7)
        expected_trace.append_operation(5, 8)
        expected_trace.append_operation(6, -1)
        expected_trace.append_operation(7, -1)
        expected_trace.append_operation(8, -1)

        # compute actual trace
        actual_trace = ted.standard_ted_backtrace(x, x_adj, y, y_adj)
        # check result
        self.assertEqual(expected_trace, actual_trace)

    def test_standard_ted_backtrace_large_scale(self):
        x_nodes, x_adj = tree_utils.from_json('minipalindrome/ImplAAA1.json')
        y_nodes, y_adj = tree_utils.from_json('minipalindrome/ImplAAB1.json')

        expected_d = ted.standard_ted(x_nodes, x_adj, y_nodes, y_adj)
        trace = ted.standard_ted_backtrace(x_nodes, x_adj, y_nodes, y_adj)

        actual_d = 0
        for op in trace:
            if(op._left >= 0 and op._right >= 0 and x_nodes[op._left] == y_nodes[op._right]):
                continue
            actual_d += 1
        # check distance
        self.assertEqual(expected_d, actual_d)

        # check if the trace fulfills all mapping constraints, i.e.
        # all nodes occur in ascending order and descendant relationships
        # between mapped nodes hold.
        i_expected = 0
        j_expected = 0
        reps = {}
        for op in trace:
            if(op._left >= 0):
                self.assertEqual(i_expected, op._left)
                i_expected += 1
                if(op._right >= 0):
                    reps[op._left] = op._right
            if(op._right >= 0):
                self.assertEqual(j_expected, op._right)
                j_expected += 1
        self.assertEqual(i_expected, len(x_nodes))
        self.assertEqual(j_expected, len(y_nodes))
        # now, verify the descendant relationships
        x_par = tree_utils.parents(x_adj)
        y_par = tree_utils.parents(y_adj)
        for i in reps:
            for k in reps:
                if(i == k):
                    continue
                self.assertEqual(self.is_ancestor(x_par, i, k), self.is_ancestor(y_par, reps[i], reps[k]), '%d [%s] should be ancestor of %d [%s] if and only if %d [%s] is ancestor of %d [%s]' % (i, x_nodes[i], k, x_nodes[k], reps[i], y_nodes[reps[i]], reps[k], y_nodes[reps[k]]))

    def is_ancestor(self, par, i, k):
        # follow the parent pointers from k up until we reach i or the root
        while(k >= 0):
            if(k == i):
                return True
            k = par[k]
        return False

    def test_speed(self):
        m = 300
        # create a very large tree with maximum number of keyroots
        x_nodes = ['a'] * (2 * m + 1)
        x_adj = []
        for i in range(m):
            x_adj.append([(i+1)*2-1, (i+1)*2])
            x_adj.append([])
        x_adj.append([])

        # as character distance, use the kronecker delta
        def kron_distance(x, y):
            if(x == y):
                return 0.
            else:
                return 1.

        # compare how long the tree edit distance takes
        start = time.time()
        d = ted.ted(x_nodes, x_adj, x_nodes, x_adj, kron_distance)
        general_time = time.time() - start


        start = time.time()
        d = ted.standard_ted(x_nodes, x_adj, x_nodes, x_adj)
        std_time = time.time() - start

        self.assertTrue(std_time < general_time)


if __name__ == '__main__':
    unittest.main()
