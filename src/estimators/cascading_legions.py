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
"""Cascading Legions cardinality estimator prototype."""

import copy
import math

import farmhash
import numpy
from scipy import special

from wfa_cardinality_estimation_evaluation_framework.estimators import base
from wfa_cardinality_estimation_evaluation_framework.estimators import bloom_filters


class CascadingLegions(base.SketchBase):
  """CascadingLegions sketch."""

  MEMOIZED_CARDINALITY = {}

  def __init__(self, l, m, random_seed):
    """Initializes the instance.

    Args:
      l: Number of legions.
      m: Number of positions in each legion.
      random_seed: A random seed for the hash function.
    """
    self.seed = random_seed
    self.l = l
    self.m = m
    self.mask = {}
    self.sketch = {}
    self.added_noise = 0
    self.get_cardinality_for_legionaries_count = bloom_filters.invert_monotonic(
        self.legions_expectation)

  @classmethod
  def get_sketch_factory(cls, l, m):
    def factory(random_seed):
      return cls(l, m, random_seed)
    return factory

  def get_bucket(self, f):
    """Computes legionary register for fingerprint f."""
    legion = 0
    while f % 2 == 0:
      legion += 1
      f //= 2
    legion = min(legion, self.l - 1)
    f //= 2
    return legion * self.m + f % self.m

  def add_fingerprint(self, f):
    """Add fingerprint to the estimator."""
    b = self.get_bucket(f)
    # This is simulation of CascadingLegions same-key-aggregator
    # frequency estimator behavior.
    self.mask[b] = self.mask.get(b, set()) | {f}
    self.sketch[b] = self.sketch.get(b, 0) + 1

  def add_id(self, item):
    """Add id to the estimator."""
    f = farmhash.hash32withseed(str(item), self.seed)
    self.add_fingerprint(f)

  def add_ids(self, item_iterable):
    """Add multiple ids to the estimator."""
    for x in item_iterable:
      self.add_id(x)

  def legions_expectation(self, cardinality):
    """Expected number of legionaries activated for cardinality."""
    cache_key = (self.l, self.m, cardinality)
    if cache_key not in self.MEMOIZED_CARDINALITY:
      r = 0
      l = 0
      for l in range(1, self.l):
        r += self.m * (1 - math.exp(-cardinality / (2 ** l * self.m)))
      r += self.m * (1 - math.exp(-cardinality / (2 ** l * self.m)))
      self.MEMOIZED_CARDINALITY[cache_key] = r
    return self.MEMOIZED_CARDINALITY[cache_key]

  def add_dp_noise(self, p):
    """Adding noise via flipping each bit with probability p."""
    if self.added_noise:
      assert False, 'Noise can only be added once. Already have noise: %.3f' % (
          self.added_noise)
    def flip(x):
      if x > 0:
        return 0
      return 1
    for i in range(self.l * self.m):
      if numpy.random.uniform(0, 1) < p:
        self.sketch[i] = flip(self.sketch.get(i, 0))
    self.added_noise = p

  def legionaries_count(self):
    """Count of legionaries in the sketch."""
    return len(self.sketch)

  def get_cardinality(self):
    """Estimating cardinality from the sketch."""
    return self.get_cardinality_for_legionaries_count(
        self.legionaries_count())

  def merge_in(self, legions):
    """Merges in another CascadingLegions."""
    assert legions.l == self.l
    assert legions.m == self.m
    for l in legions.sketch:
      self.sketch[l] = self.sketch.get(l, 0) + legions.sketch[l]
      self.mask[l] = self.mask.get(l, set()) | legions.mask.get(l)

  def frequency_sample(self):
    """Sample of frequencies."""
    sample = []
    for l in self.sketch:
      if len(self.mask[l]) == 1:
        sample.append(self.sketch[l])
    return sample

  def frequency_histogram(self):
    """Estimated frequency histogram."""
    result = {}
    for x in self.frequency_sample():
      result[x] = result.get(x, 0) + 1
    total = sum(result.values())
    for x in result:
      result[x] /= total
    return result


class Noiser(base.SketchNoiserBase):
  """Noiser of CascadingLegions."""

  def __init__(self, flip_probability):
    self.flip_probability = flip_probability

  def __call__(self, sketch):
    sketch_copy = copy.deepcopy(sketch)
    sketch_copy.add_dp_noise(self.flip_probability)
    return sketch_copy


class Estimator(base.EstimatorBase):
  """Estimator for DP-noised CascadingLegions."""

  def __init__(self, flip_probability=None):
    self.flip_probability = flip_probability

  def __call__(self, sketch_list):
    """Estimating cardinality of the union."""
    if not sketch_list:
      return 0  # That was easy!

    flip_probability = self.flip_probability or sketch_list[0].added_noise

    sketch_noises = {s.added_noise for s in sketch_list}
    assert sketch_noises == {
        flip_probability
    }, (f'Sketches have inconsistent noise. Actual: {sketch_noises}, but '
        f'should all be equal to {flip_probability}.')
    cardinality, unused_golden_legion_index = self.estimate_from_golden_legion(
        sketch_list, flip_probability)
    return cardinality

  @classmethod
  def legion_as_vector(cls, sketch_list, legion_index):
    """Legion as a vector of counts of positions with i ones.

    E.g. if legions are
    10110
    10001

    Then the vector is:
    1,3,1
    I.e. 1 position with 0 ones, 3 positions with 1 one, 1 position with 2 ones.

    Args:
      sketch_list: A list of CascadingLegions.
      legion_index: Index of the legion to count.

    Returns:
      A vector, which i-th element is equal to the count of positions in the
        legion legion_index where exactly i sketches have 1.
    """
    n = sketch_list[0].m
    v = [0] * (len(sketch_list) + 1)
    for i in range(n):
      x = 0
      for s in sketch_list:
        x += (s.sketch.get(legion_index * n + i, 0) > 0)
      v[x] += 1
    return v

  @classmethod
  def transition_probability(cls, num_sketches, s, t, p):
    """Probability of vector of counts transition.

    Args:
      num_sketches: Number of sketches.
      s: Transitioning from s ones.
      t: Transitioning to t ones.
      p: Probability of flip.

    Returns:
      Probability of a position of num_sketches transition from s number of
      ones to t number of ones if each bit is flipped with probability p.
    """
    q = 1 - p
    result = 0
    for i in range(num_sketches // 2 + 1):
      flip_zeros = max(0, t - s) + i
      flip_ones = max(0, s - t) + i
      flips = flip_ones + flip_zeros
      calms = num_sketches - flips
      if flip_ones > s or flip_zeros > num_sketches - s:
        continue
      choose_ones = special.comb(s, flip_ones)
      choose_zeros = special.comb(num_sketches - s, flip_zeros)
      choices = choose_ones * choose_zeros
      assert choices > 0
      result += choices * p ** flips * q ** calms
    return result

  @classmethod
  def transition_matrix(cls, num_sketches, p):
    """Stochastic matrix of transitions of counts of ones."""
    result = []
    for row_index in range(num_sketches + 1):
      row = []
      for column_index in range(num_sketches + 1):
        row.append(cls.transition_probability(
            num_sketches, column_index, row_index, p))
      result.append(row)
    return numpy.array(result)

  @classmethod
  def correction_matrix(cls, num_sketches, p):
    """Matrix to denoise the vector of counts."""
    return numpy.linalg.inv(cls.transition_matrix(num_sketches, p))

  @classmethod
  def estimate_from_one_legion(cls, sketch_list, legion_index, p):
    """Linear time estimator of cardinality from a noised Legion."""
    c = cls.correction_matrix(len(sketch_list), p)
    v = cls.legion_as_vector(sketch_list, legion_index)
    f = sum(v) - c[0, :].dot(v)
    n = sketch_list[0].m
    if f > n:
      return 2 ** legion_index * n * 10
    return -math.log(1 - f / n) * n * (2 ** (legion_index + 1))

  @classmethod
  def estimate_from_golden_legion(cls, sketch_list, p):
    """Estimate cardinality from Golden Legion."""
    l = sketch_list[0].l
    n = sketch_list[0].m
    for i in range(l):
      e = cls.estimate_from_one_legion(sketch_list, i, p)
      # i-th legion does sampling of 1 in 2 ** (i + 1). We declare legion
      # oversaturated if has more tha n / 2 items it in.
      if e < n / 2 * 2 ** (i + 1):
        return e, i
    assert False, (
        f'Not enough legions to estimate. I have {l} legions, but the '
        f'cardinality appears to be greater than {e}.')
