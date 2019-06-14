# Adversarial Edit Attacks

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

## Introduction

This is a Python 3 implementation of adversarial edit attacks on tree structured
data, as presented in the paper

* Paaßen, B. (2019). Adversarial Edit Attacks for Tree-structured Data.
    Submitted to the ECML PKDD Joint International Workshop on
    Advances in Interpretable Machine Learning and Artificial Intelligence &
    Explainable Knowledge Discovery in Data Mining (AIMLAI-XKDD 2019)

The attacks presented here manipulate an input tree _x_ with tree edits until
the predicted label of an attacked tree classifier _f_ changes, i.e. the
adversarial tree _z_ is such that _f(x) &ne; f(z)_. As additional requirement,
we try to apply as few tree edits as possible, i.e. we try to minimize the
tree edit distance _d(x, z)_.

In more detail, this software package implements two attacks, one which applies
random tree edits until the label changes, and one which applies tree edits
to reduce the distance to a reference tree which has the desired target label
via backtracing ([Paaßen, 2018][1]).
These methods are implemented in `adversarial_edits.construct_random_adversarial`
and `adversarial_edits.construct_adversarial`, respectively.

## Reproduce the results in the paper

To reproduce the results presented in the paper, you merely have to run the
jupyter notebooks `MiniPalindrome.ipynb`, `Sorting.ipynb`, `Cystic.ipynb`,
and `Leukemia.ipynb`. Afterwards, the `Results.ipynb` performs the visualization
as a LaTeX table and the statistical tests.

Note that you need certain dependencies to run the notebooks. Please refer to
the Dependencies section below for more information.

## Attack your own data

To use the code here to attack novel data, you can use the
`adversarial_edits.construct_random_adversarials` method (for random attacks)
or the `adversarial_edits.construct_adversarials` method (for backtracing
attacks). Please refer to the function documentation of these methods for more
details. Note that the latter method requires a pairwise tree edit distance
matrix on the data for which adversarials should be constructed. This pairwise
tree edit distance matrix can be computed via

<pre>
import multiprocess
D = multiprocess.pairwise_distances_symmetric(X, X)
</pre>

## Dependencies

As dependencies, this package requires [numpy](http://www.numpy.org/) for
general array handling, [scipy](https://scipy.org/) for eigenvalue decomposition
and statistical tests, [pytorch](https://pytorch.org/) for recursive neural
networks, [cython](https://cython.org/) for fast tree edit distance
computations, and [ptk](http://www.joedsm.altervista.org/pythontreekernels.htm)
for tree kernel computations. Note that the latter package is not available via
pip and is written in Python2, such that we include an adapted Python3 version
here in the subfolder `ptk`.

## License

This documentation is licensed under the terms of the [creative commons attribution-shareAlike 4.0 international (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/) license. The code
contained alongside this documentation is licensed under the
[GNU General Public License Version 3](https://www.gnu.org/licenses/gpl-3.0.en.html) license.
A copy of this license is contained in the `gpl-3.0.md` file alongside this README.

## Literature

* Paaßen, B. (2018). Revisiting the tree edit distance and its backtracing: A tutorial. [arXiv:1805.06869][1]

[1]: https://arxiv.org/abs/1805.06869 "Paaßen, B. (2018). Revisiting the tree edit distance and its backtracing: A tutorial. arXiv:1805.06869"
