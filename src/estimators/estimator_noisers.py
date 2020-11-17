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

"""Noisers for sketches."""

import numpy as np
from wfa_cardinality_estimation_evaluation_framework.common import noisers
from wfa_cardinality_estimation_evaluation_framework.estimators import base



class LaplaceEstimateNoiser(base.EstimateNoiserBase):
  """A noiser that adds Laplace noise to a cardinality estimate."""

  def __init__(self, epsilon, random_state=None):
    """Instantiates a LaplaceEstimateNoiser object.

    Args:
      epsilon:  The differential privacy level.
      random_state:  Optional instance of numpy.random.RandomState that is used
        to seed the random number generator.
    """
    # Note that any cardinality estimator will have sensitivity (delta_f) of 1.
    self._noiser = noisers.LaplaceMechanism(lambda x: x, 1.0, epsilon,
                                            random_state=random_state)

  def __call__(self, cardinality_estimate):
    """Returns a cardinality estimate with Laplace noise."""
    if type(cardinality_estimate) == float:
      return self._noiser(np.array([cardinality_estimate]))[0]
    else:
      return self._noiser(cardinality_estimate)


class GeometricEstimateNoiser(base.EstimateNoiserBase):
  """A noiser that adds discrete Laplace noise to a cardinality estimate."""

  def __init__(self, epsilon, random_state=None):
    """Instantiates a GeometricEstimateNoiser object.

    Args:
      epsilon:  The differential privacy level.
      random_state:  Optional instance of numpy.random.RandomState that is used
        to seed the random number generator.
    """
    # Note that any cardinality estimator will have sensitivity (delta_f) of 1.
    self._noiser = noisers.GeometricMechanism(lambda x: x, 1.0, epsilon,
                                              random_state=random_state)

  def __call__(self, cardinality_estimate):
    """Returns a cardinality estimate with discrete Laplace noise."""
    if type(cardinality_estimate) == float:
      return self._noiser(np.array([cardinality_estimate]))[0]
    else:
      return self._noiser(cardinality_estimate)


class GaussianEstimateNoiser(base.EstimateNoiserBase):
  """A noiser that adds Gaussian noise to a cardinality estimate."""

  def __init__(self, epsilon, delta, num_queries=1, random_state=None):
    """Instantiates a GaussianEstimateNoiser object.

    Args:
      epsilon:  The differential privacy level.
      delta:  The differential privacy level.
      num_queries: The number of queries for which the noiser is used. Note
        that the constructed noiser will be (epsilon, delta)-differentially
        private when answering (no more than) num_queries queries.
      random_state:  Optional instance of numpy.random.RandomState that is used
        to seed the random number generator.
    """
    # Note that any cardinality estimator will have sensitivity (delta_f) of 1.
    self._noiser = noisers.GaussianMechanism(
      lambda x: x, 1.0, epsilon, delta, num_queries=num_queries,
      random_state=random_state)

  def __call__(self, cardinality_estimate):
    """Returns a cardinality estimate with Gaussian noise."""
    if type(cardinality_estimate) == float:
      return self._noiser(np.array([cardinality_estimate]))[0]
    else:
      return self._noiser(cardinality_estimate)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')


if __name__ == '__main__':
  app.run(main)
