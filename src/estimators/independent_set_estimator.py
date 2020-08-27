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
"""Estimator for sketches when using the independence assumption.

Under the independence assumption, the cardinality of the union of two
sketches representing sets A and B is equal to

  |A| + |B| - |A| * |B| / N,

where N is the size of the universe from which A and B are drawn.
"""
import copy
from itertools import accumulate
from wfa_cardinality_estimation_evaluation_framework.common import noisers
from wfa_cardinality_estimation_evaluation_framework.estimators.base import EstimatorBase
from wfa_cardinality_estimation_evaluation_framework.estimators.base import SketchBase
from wfa_cardinality_estimation_evaluation_framework.estimators.base import SketchNoiserBase
from wfa_cardinality_estimation_evaluation_framework.estimators.exact_set import ExactMultiSet
from wfa_cardinality_estimation_evaluation_framework.estimators.exact_set import LosslessEstimator


class Reach(SketchBase):
  """Reach of a campaign."""

  @classmethod
  def get_sketch_factory(cls):

    def f(random_seed):
      return cls(random_seed=random_seed)

    return f

  def __init__(self, random_seed=None):
    """Construct reach sketch.

    Args:
      random_seed: This arg exists in order to conform to
        simulator.EstimatorConfig.sketch_factory.
    """
    SketchBase.__init__(self)
    self._cardinality = 0
    self._estimator = LosslessEstimator()
    self._exact_set = ExactMultiSet(random_seed)

  def add_ids(self, ids):
    for x in ids:
      self._exact_set.add(x)
    self._cardinality = self._estimator([self._exact_set])

  @property
  def cardinality(self):
    return self._cardinality

  @cardinality.setter
  def cardinality(self, new_cardinality):
    assert len(new_cardinality) == len(self.cardinality)
    for i, x in enumerate(new_cardinality):
      self._cardinality[i] = max(0, new_cardinality[i])


class ReachEstimator(EstimatorBase):
  """Reach of a single campaign."""

  def __call__(self, sketch_list):
    """Get the reach."""
    assert len(sketch_list) == 1, 'The input should be a list of one element.'
    return sketch_list[0].cardinality


class ReachNoiser(SketchNoiserBase):
  """Reach noiser."""

  def __init__(self, epsilon, random_state=None):
    """Instantiates the noiser object.

    Args:
      epsilon: The differential privacy level.
      random_state:  Optional instance of numpy.random.RandomState that is used
        to seed the random number generator.
    """
    self._noiser = noisers.GeometricMechanism(lambda x: x, 1.0, epsilon,
                                              random_state)

  def __call__(self, sketch):
    """Returns a noised reach sketch."""
    new_sketch = copy.deepcopy(sketch)
    new_sketch.cardinality = [self._noiser(x) for x in sketch.cardinality]
    return new_sketch


class IndependentSetEstimator(EstimatorBase):
  """An estimator that estimates the size of a union by assuming independence.

  Given two sketches A and B representing samples drawn from a universe U,
  the estimated size of A union B is given by:

    |A union B| = |A| + |B| - |A| * |B| / |U|.

  Note that the estimators for A and B will in general return cumulative histograms,
  which may contain values for frequencies greater than 1.  The
  IndependentSetEstimator also returns a cumulative histogram, which estimates the
  number of items in the intersection at various frequencies.
  """

  def __init__(self, single_sketch_estimator, universe_size):
    """Instantiates an IndependentSetEstimator object.

    Args:
      single_sketch_estimator:  An object derived from EstimatorBase for
        providing cardinality estimates of individual sketches.  The
        single_sketch_estimator is assumed to return a cumulative histogram.
      universe_size: Size of the universe.
    """
    self.single_sketch_estimator = single_sketch_estimator
    self.universe_size = universe_size

  def __call__(self, sketch_list):
    """Computes the frequency histogram of a union of sketches.

    Args:
      sketch_list: A list of sketches.  For each sketch in the list, its
        cumulative frequency histogram is determined using the 
        single_sketch_estimator that was provided in the 
        IndependentSetEstimator constructor.

    Returns:
      A cumulative histogram h.  The element h[i] represents the number
      of items in the union having frequency i+1 or greater, under the
      independence assumption.
    """
    if not sketch_list:
      return [0]
    
    a_hist = [0]  # Exact frequencies of current estimate
    for sketch in sketch_list:
      # ch is a cumulative histogram.
      ch = self.single_sketch_estimator([sketch])
      # b_hist is the exact frequency histogram for the new sketch.
      b_hist = [ch[i] - ch[i+1] for i in range(len(ch)-1)] + [ch[-1]]

      # To compute the histogram of the union, we look at all pairs
      # A[i] and B[j], where A[i] (resp. B[j]) is the number of items with
      # frequency i+1 in A (resp. j+1 in B).  Items in the intersection
      # of A[i] and B[j] will have frequency i+j+2 in the union.
      # These items will need to be deducted from the frequency counts for
      # both i and j to avoid overcounting.
      c_hist = a_hist.copy() + [0] * (len(b_hist) + 1)
      for i in range(len(b_hist)):
        c_hist[i] += b_hist[i]
      for i in range(len(a_hist)):
        for j in range(len(b_hist)):
          overlap = a_hist[i] * b_hist[j] / float(self.universe_size)
          if overlap:
            c_hist[i] -= overlap
            c_hist[j] -= overlap
            c_hist[i+j+1] += overlap

      a_hist = c_hist

      assert sum(a_hist) <= self.universe_size, (
        "Constraint violation: sketch is larger than universe")

    # Trim away trailing 0's
    while len(a_hist) > 0 and a_hist[-1] == 0:
      del a_hist[-1]
      
    # At this point, a_hist is the histogram of exact frequencies,
    # e.g., a_hist[i] is the number of items having frequency exactly i+1.
    # Now convert it into a histogram of cumulative frequencies.

    cumulative_hist = list(reversed(list(accumulate(reversed(a_hist)))))

    return cumulative_hist
