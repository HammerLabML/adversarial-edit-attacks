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

## Installation and setup

To set up this package, you need to

1. install all dependencies listed below (except for `ptk`, which is enclosed),
2. run the command `python3 setup.py build_ext --inplace` to compile the cython
    sources.

Then, every function should run.

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
and statistical tests, [sklearn](https://scikit-learn.org/stable/) for
support vector machines, [pytorch](https://pytorch.org/) for recursive neural
networks, [cython](https://cython.org/) for fast tree edit distance
computations, and [ptk][2]
for tree kernel computations. Note that the latter package is not available via
pip and is written in Python2, such that we include an adapted Python3 version
here in the subfolder `ptk`.

## License

This documentation is licensed under the terms of the [creative commons attribution-shareAlike 4.0 international (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/) license. The code
contained alongside this documentation is licensed under the
[GNU General Public License Version 3](https://www.gnu.org/licenses/gpl-3.0.en.html) license.
A copy of this license is contained in the `gpl-3.0.md` file alongside this README.

## Contents

The detailed contents of this package are the following:

* `adversarial_edits.py` : Implements adversarial edit attacks.
* `adversarial_edits_test.py` : Provides test functions for `adversarial_edits.py`.
* `cystic` : Contains the Cystic data set in JSON format.
* `Cystic.ipynb` : Contains the Cystic experiment.
* `gpl-3.0.md` : Contains the GPLv3 license.
* `hyperopt.py` : Implements hyper parameter optimization for SVM and tree echo
    state networks.
* `hyperopt_test.py` : Provides test functions for `hyperopt.py`.
* `leukemia` : Contains the Leukemia data set in JSON format.
* `Leukemia.ipynb` : Contains the Leukemia experiment.
* `minipalindrome` : Contains the MiniPalindrome data set in JSON format.
* `MiniPalindrome.ipynb` : Contains the MiniPalindrome experiment.
* `ptk` : Contains a python3 compatible version of Giovanni Da San Martino's
          [python tree kernel (ptk) toolbox][2].
* `ptk_utils.py` : Contains interface functions for the ptk toolbox.
* `README.md` : This file.
* `recursive_net.py` : Implements recursive neural networks
    ([Sperduti & Starita, 1997][3]) in [pytorch](https://pytorch.org/).
* `recursive_net_test.py` : Provides test functions for `recursive_net.py`.
* `results` : Contains experimental results.
* `Resulty.ipynb` : Evaluates the experimental results.
* `setup.py` : A helper script to compile the `ted.pyx` file using
    [cython](https://cython.org/).
* `sorting` : Contains the Sorting data set in JSON format.
* `Sorting.ipynb` : Contains the Sorting experiment.
* `ted.pyx` : Implements the tree edit distance and its backtracing following
    [Paaßen (2018)][1].
* `ted_test.py` : Provides test functions for `ted.pyx`.
* `trace.py` : Contains utility classes for tree edit distance backtracing.
* `tree_echo_state.py` : Implements Tree Echo State nwtorks
    ([Gallicchio & Micheli, 2013][4]).
* `tree_echo_state_test.py` : Provides test functions for `tree_echo_state.py`.
* `tree_edits.py` : Implements tree edits as described in the paper.
* `tree_edits_test.py` : Provides test functions for `tree_edits.py`.
* `tree_utils.py` : Provides utility functions for tree processing.
* `tree_utils_test.py` : Provides test functions for `tree_utils.py`.

## Literature

* Paaßen, B. (2018). Revisiting the tree edit distance and its backtracing: A tutorial. [arXiv:1805.06869][1]
* Sperduti, A. & Starita, A. (1997). Supervised neural networks for the classification of structures. IEEE Transactions on Neural Networks, 8(3), 714-735. doi:[10.1109/72.572108][3]
* Gallicchio, C. & Micheli, A. (2013). Tree Echo State Networks. Neurocomputing, 101, 319-337. doi:[10.1016/j.neucom.2012.08.017][4]

[1]: https://arxiv.org/abs/1805.06869 "Paaßen, B. (2018). Revisiting the tree edit distance and its backtracing: A tutorial. arXiv:1805.06869."
[2]: http://www.joedsm.altervista.org/pythontreekernels.htm "Python tree kernels, as provided by Giovanni da san Martino."
[3]: http://doi.org/10.1109/72.572108 "Sperduti, A. & Starita, A. (1997). Supervised neural networks for the classification of structures. IEEE Transactions on Neural Networks, 8(3), 714-735."
[4]: http://doi.org/10.1016/j.neucom.2012.08.017 "Gallicchio, C. & Micheli, A. (2013). Tree Echo State Networks. Neurocomputing, 101, 319-337."
