# Copyright 2020 The Private Cardinality Estimation Framework Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import numpy as np
from dp_accounting import accountant
from dp_accounting import common


class LaplaceMechanism:
  """Transforms a function using the Laplace mechanism.

  If f(x) = (Z[1], Z[2], ..., Z[k]), then returns a function that computes
  (Z'[1], Z'[2], ..., Z'[k]), where
      Z'[i] = Z[i] + Y[i],
      Y[i] ~ Lap(x | delta_f / epsilon),
  and Lap(x | b) is given by the probability density function
      Lap(x | b) = (1 / 2b) exp(-|x| / b).

  See section 3.3 of Dwork and Roth.
  """

  def __init__(self, f, delta_f, epsilon, random_state=None):
    """Instantiates a LaplaceMechanism.

    Args:
      f: A function which takes as input a database and which returns as output
        a numpy array.
      delta_f: The sensitivity paramater, e.g., the maximum value by which the
        function can change for two databases that differ by only one row.
      epsilon: Differential privacy parameter.
      random_state:  Optional instance of numpy.random.RandomState that is
        used to seed the random number generator.
    """
    self._func = f
    self._delta_f = delta_f
    self._epsilon = epsilon
    self._random_state = random_state or np.random.RandomState()

  def __call__(self, x):
    z = self._func(x)
    return z + self._random_state.laplace(
        size=z.shape, scale=self._delta_f / self._epsilon)


class GeometricMechanism:
  """Transforms a function using the geometric mechanism.

  If f(x) = (Z[1], Z[2], ..., Z[k]), then returns a function that computes
  (Z'[1], Z'[2], ..., Z'[k]), where
      Z'[i] = Z[i] + Y[i],
      Y[i] ~ DL(k | exp(-epsilon / delta_f)),
  and DL(k | alpha) is a probability mass function defined on the
  integers that is given by
      DL(k | alpha) = (1 - alpha) / (1 + alpha) * alpha ^ |k|

  DL(k | alpha) is sometimes referred to as the discrete Laplace
  distribution.  See:

  Inusah, Seidu, and Tomasz J. Kozubowski. "A discrete analogue of the
  Laplace distribution." Journal of statistical planning and inference
  136.3 (2006): 1090-1102.

  The geometric mechanism was defined in:

  Ghosh, Arpita, Tim Roughgarden, and Mukund Sundararajan.
  "Universally utility-maximizing privacy mechanisms."
  SIAM Journal on Computing 41.6 (2012): 1673-1693.

  The geometric mechanism should not be confused with the geometric
  distribution.  The geometric distribution has PMF
    Pr(X=k | p) = p * (1-p)^k-1.
  There is a connection between the geometric distribution and the discrete
  Laplace distribution, though.  If X and Y are independent random variables
  having geometric distribution p, then X-Y is a discrete Laplace random
  variable with parameter 1-p.
  """

  def __init__(self, f, delta_f, epsilon, random_state=None):
    """Instantiates a geometric mechanism.

    Args:
      f: A function which takes as input a database and which returns as output
        a numpy array.
      delta_f: The sensitivity paramater, e.g., the maximum value by which the
        function can change for two databases that differ by only one row.
      epsilon: Differential privacy parameter.
      random_state:  Optional instance of numpy.random.RandomState that is
        used to seed the random number generator.
    """
    self._func = f
    self._delta_f = delta_f
    self._epsilon = epsilon
    self._random_state = random_state or np.random.RandomState()

  def __call__(self, x):
    z = self._func(x)
    p_geometric = 1 - math.exp(-self._epsilon / self._delta_f)
    x = self._random_state.geometric(size=z.shape, p=p_geometric)
    y = self._random_state.geometric(size=z.shape, p=p_geometric)
    return z + x - y


class GaussianMechanism:
  """Transforms a function using the gaussian mechanism.

  If f(x) = (Z[1], Z[2], ..., Z[k]), then returns a function that computes
  (Z'[1], Z'[2], ..., Z'[k]), where
      Z'[i] = Z[i] + Y[i],
      Y[i] ~ N(x | sigma),
  and N(x | sigma) is given by the probability density function
      N(x | sigma) = exp(-0.5 x^2 / sigma^2) / (sigma * sqrt(2 * pi))

  See Appendix A of Dwork and Roth.
  """

  def __init__(
    self, f, delta_f, epsilon, delta, num_queries=1, random_state=None):
    """Instantiates a gaussian mechanism.

    Args:
      f: A function which takes as input a database and which returns as output
        a numpy array.
      delta_f: The sensitivity paramater, e.g., the maximum value by which the
        function can change for two databases that differ by only one row.
      epsilon: Differential privacy parameter.
      delta: Differential privacy parameter.
      num_queries: The number of queries for which the mechanism is used. Note
        that the constructed mechanism will be (epsilon, delta)-differentially
        private when answering (no more than) num_queries queries.
      random_state:  Optional instance of numpy.random.RandomState that is
        used to seed the random number generator.
    """
    self._func = f
    self._delta_f = delta_f
    self._sigma = accountant.get_smallest_gaussian_noise(
      common.DifferentialPrivacyParameters(epsilon, delta),
      num_queries, sensitivity=delta_f)
    self._random_state = random_state or np.random.RandomState()

  def __call__(self, x):
    z = self._func(x)
    return z + self._random_state.normal(size=z.shape, scale=self._sigma)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')


if __name__ == '__main__':
  app.run(main)
