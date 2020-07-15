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

"""Simple example of creating a cardinality estimator."""

import copy
import sys
from wfa_cardinality_estimation_evaluation_framework.estimators.base import EstimatorBase
from wfa_cardinality_estimation_evaluation_framework.estimators.base import SketchNoiserBase
from wfa_cardinality_estimation_evaluation_framework.estimators.base import SketchBase


class ExactSet(SketchBase):
  """An sketch that just uses a set as its implementation."""

  @classmethod
  def get_sketch_factory(cls):

    def f(random_seed):
      return cls(random_seed=random_seed)

    return f

  def __init__(self, random_seed=None):
    """Construct an ExactSet sketch.

    Args:
      random_seed: This arg exists in order to conform to
        simulator.EstimatorConfig.sketch_factory.
    """
    SketchBase.__init__(self)
    _ = random_seed
    self._ids = set()

  def __len__(self):
    """Return the number of elements in the sketch."""
    return len(self._ids)

  def __contains__(self, x):
    """Return true if x is contained in the sketch."""
    return x in self._ids

  def add(self, x):
    """Adds an id x to the sketch."""
    self._ids.add(x)

  def ids(self):
    """Return the internal ID set."""
    return self._ids


class LosslessEstimator(EstimatorBase):
  """A lossless estimator for ExactSet."""

  def __init__(self):
    EstimatorBase.__init__(self)

  def __call__(self, sketch_list):
    """Return len(sketch)."""
    union = ExactSet()
    for s in sketch_list:
      union.add_ids(s.ids())
    return [len(union)]


class LessOneEstimator(EstimatorBase):
  """An estimator for ExactSet that is always short by one."""

  def __init__(self):
    EstimatorBase.__init__(self)

  def __call__(self, sketch_list):
    """Return len(sketch)."""
    e = LosslessEstimator()
    histogram = e(sketch_list)
    return [histogram[0]-1] + [histogram[1:]]


class AddRandomElementsNoiser(SketchNoiserBase):
  """An example Noiser."""

  def __init__(self, num_random_elements, random_state):
    SketchNoiserBase.__init__(self)
    self.num_random_elements = num_random_elements
    self.random_state = random_state

  def __call__(self, sketch):
    """Adds three random items to the sketch."""
    new_sketch = copy.deepcopy(sketch)
    num_added = 0
    while num_added < self.num_random_elements:
      x = self.random_state.randint(sys.maxsize)
      if x not in new_sketch:
        new_sketch.add_ids([x])
        num_added += 1
    return new_sketch
