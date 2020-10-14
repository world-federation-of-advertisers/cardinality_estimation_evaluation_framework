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
"""Liquid Legions cardinality estimator prototype."""

import farmhash
import numpy
from scipy import special

from wfa_cardinality_estimation_evaluation_framework.estimators import base
from wfa_cardinality_estimation_evaluation_framework.estimators import bloom_filters
from wfa_cardinality_estimation_evaluation_framework.estimators import cascading_legions


def unique_aggregator(a, b):
  """Unique Aggregator.

  If all aggregated values are the same then result of aggregation is equal to
  the value. Otherwise it is -1.
  Using None to mean that a source had no value for the key.

  Args:
    a: First value to aggregate.
    b: Second value to aggregate.

  Returns:
    result of aggregation.
  """
  if a is None or b is None or a == b:
    return a or b
  return -1


class LiquidLegions(base.SketchBase):
  """LiquidLegions sketch."""

  def __init__(self, a, m, random_seed):
    """Initializes the instance.

    Args:
      a: Rate of the exponential distribution.
      m: Number of legionaries.
      random_seed: A seed to use for hashing.
    """
    assert a > 0, f'Parameter a must be positive. {a} is not.'
    self.seed = random_seed
    self.memoized_cardinality = {}
    self.a = a
    self.m = m
    self.unique = {}
    self.sketch = {}
    self.added_noise = 0

  def sampler(self, u):
    """A function mapping uniform distribution to truncated exponential."""
    a = self.a
    return 1 - numpy.log(numpy.exp(a) + u * (1 - numpy.exp(a))) / a

  @classmethod
  def get_sketch_factory(cls, a, m):
    def factory(random_seed):
      return cls(a, m, random_seed)
    return factory

  def get_bucket(self, f):
    """Computes legionary register for fingerprint f."""
    u = f / 2 ** 32
    f = self.sampler(u)
    return int(f * self.m)

  def add_fingerprint(self, f):
    """Add fingerprint to the estimator."""
    b = self.get_bucket(f)
    # This is simulation of LiquidLegions same-key-aggregator
    # frequency estimator behavior.
    if b not in self.unique:
      self.unique[b] = f
    else:
      self.unique[b] = unique_aggregator(self.unique.get(b), f)
    self.sketch[b] = self.sketch.get(b, 0) + 1

  def add_id(self, item):
    """Add id to the estimator."""
    f = farmhash.hash32withseed(str(item), self.seed)
    self.add_fingerprint(f)

  def add_ids(self, items):
    """Add a list of ids to the estimator."""
    for item in items:
      self.add_id(item)

  def legions_expectation(self, t):
    """Expected number of legionaries activated for cardinality."""
    cache_key = (self.a, self.m, t)
    if cache_key not in self.memoized_cardinality:
      if t == 0:
        self.memoized_cardinality[cache_key] = 0
      else:
        a = self.a
        z = 1 - (-special.expi(- a * t / (numpy.exp(a) - 1))
                +special.expi(-a * numpy.exp(a) * t / (numpy.exp(a) - 1))) / a
        r = numpy.where(t == 0, 0, z)
        self.memoized_cardinality[cache_key] = r
      return self.memoized_cardinality[cache_key]

  def add_dp_noise(self, p):
    """Adding noise via flipping each bit with probability p."""
    if self.added_noise:
      assert False, 'Noise can only be added once. Already have noise: %.3f' % (
          self.added_noise)
    def flip(x):
      if x > 0:
        return 0
      return 1
    for i in range(self.m):
      if numpy.random.uniform(0, 1) < p:
        self.sketch[i] = flip(self.sketch.get(i, 0))
    self.added_noise = p

  def legionaries_count(self):
    """Count of legionaries in the sketch."""
    return len(self.sketch)

  def get_relative_cardinality(self, p):
    """Estimate the relative cardinality for a given count of legionaries."""
    return bloom_filters.invert_monotonic(
        self.legions_expectation, epsilon=0.0000001)(p)

  def get_cardinality_for_legionaries_count(self, count):
    """Estimates cardinality for a given count of legionaries."""
    def clip(x, a, b):
      return max(min(x, b), a)
    p = clip(count / self.m, 0, self.m - 1)
    result = self.get_relative_cardinality(p) * self.m
    assert result >= 0, 'This should never happen.'
    return result

  def get_cardinality(self):
    """Estimating cardinality from the sketch."""
    return self.get_cardinality_for_legionaries_count(self.legionaries_count())

  def merge_in(self, other_sketch):
    """Merges in another Liquidother_sketch."""
    assert other_sketch.a == self.a
    assert other_sketch.m == self.m
    for l in other_sketch.sketch:
      self.sketch[l] = self.sketch.get(l, 0) + other_sketch.sketch[l]
      self.unique[l] = unique_aggregator(self.unique.get(l),
                                         other_sketch.unique.get(l))

  @classmethod
  def merge_of(cls, sketches):
    """Merges all clear sketches."""
    assert sketches, 'I do not merge empty sets.'
    result = sketches[0].get_compatible_sketch()
    for s in sketches:
      result.merge_in(s)
    return result

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

  def pdf(self, x):
    """Probabily density function at x."""
    return self.a * numpy.exp(-self.a * x) / (1 - numpy.exp(-self.a))

  def hit_probability(self, index, cardinality):
    """Probability that a register gets hit at cardinality."""
    return 1 - numpy.exp(-self.pdf(index / self.m) / self.m * cardinality)

  def get_compatible_sketch(self):
    """Returns a sketch compatible (mergeble) with self."""
    return LiquidLegions(a=self.a, m=self.m, random_seed=self.seed)


class Noiser(cascading_legions.Noiser):
  """Noiser of LiquidLegions."""

  # Identical to CascadingLegions' noiser.
  pass


class Estimator(base.EstimatorBase):
  """Estimator for DP-noised LiquidLegions."""

  def __init__(self, flip_probability=None):
    self.flip_probability = flip_probability

  def __call__(self, sketch_list):
    """Estimating cardinality of the union."""
    if not sketch_list:
      return 0  # That was easy!

    flip_probability = self.flip_probability or sketch_list[0].added_noise

    sketch_noises = {s.added_noise for s in sketch_list}
    assert sketch_noises == {flip_probability}, (
        f'Sketches have inconsistent noise. Actual: {sketch_noises}, but '
        f'should all be equal to {flip_probability}.')
    cardinality = self.estimate_from_all(sketch_list, flip_probability)
    return [cardinality]

  @classmethod
  def sublegion_as_vector(cls, sketch_list, start, m):
    """Legion as a vector of counts of positions with i ones.

    E.g. if legions are
    10110
    10001

    Then the vector is:
    1,3,1
    I.e. 1 position with 0 ones, 3 positions with 1 one, 1 position with 2 ones.

    Args:
      sketch_list: A list of LiquidLegions.
      start: Start index of the legion to count.
      m: Size of the sublegion to count.

    Returns:
      A vector, which i-th element is equal to the count of positions in the
        sublegion (start, ..., start + m - 1) where exactly i sketches have 1.
    """
    v = [0] * (len(sketch_list) + 1)
    for i in range(m):
      x = 0
      for s in sketch_list:
        x += (s.sketch.get(start + i, 0) > 0)
      v[x] += 1
    return v

  @classmethod
  def correction_matrix(cls, num_sketches, p):
    """Matrix to denoise the vector of counts."""
    return cascading_legions.Estimator.correction_matrix(num_sketches, p)

  @classmethod
  def estimate_from_all(cls, sketch_list, p):
    """Linear time estimator of cardinality from a noised Legion."""
    m = sketch_list[0].m
    c = cls.correction_matrix(len(sketch_list), p)
    v = cls.sublegion_as_vector(sketch_list, 0, sketch_list[0].m)
    f = sum(v) - c[0, :].dot(v)
    f = max(0, min(m - 1, f))
    return sketch_list[0].get_cardinality_for_legionaries_count(f)


class VennEstimator(object):
  """Estimator of Venn diagram components' cardinalities and probabilities."""

  def __init__(self, sketch_list):
    self.num_sketches = len(sketch_list)
    self.sketch_list = sketch_list
    self.ps = [s.added_noise for s in sketch_list]
    self.sketch = self.sketch_list[0].get_compatible_sketch()

  def __call__(self):
    """Estimating cardinality of each piece of Venn diagram.

    Element number i of Venn diagram is defined as follows:
      1) Translate i to a bit vector, where j-th element corresponds to j-th bit
      of i.
      2) Take an intersection of all sets which bits are set to 1. Subtract all
      sets which bit is set to 0.
    Returns:
      A list of cardinalities, where i-th element is cardinality of i-th
      element of Venn diagram. See definition above. With the exception of 0-th
      element, which is 0.
    """
    return self.estimate_from_all()

  def sublegion_as_vector(self, start, m):
    """Vector of counts of combinations in the sublegion."""
    v = [0] * (2 ** len(self.sketch_list))
    for i in range(m):
      x = self.observation_at_index(start + i)
      v[x] += 1
    return v

  def observation_at_index(self, index):
    x = 0
    for sketch_index, s in enumerate(self.sketch_list):
      x += 2 ** sketch_index * (s.sketch.get(index, 0) > 0)
    return x

  def transition_probability(self, s, t):
    """Probability of vector of counts transition.

    Args:
      s: Transitioning combination s.
      t: Transitioning to combination t.

    Returns:
      Probability of a position of num_sketches transition from s number of
      ones to t number of ones if each bit is flipped with probability p.
    """
    result = 1
    for p in self.ps:
      result *= (1 - p) if s % 2 == t % 2 else p
      s //= 2
      t //= 2
    return result

  def transition_matrix(self):
    """Stochastic matrix of transitions of combinations of ones."""
    result = []
    for row_index in range(2 ** self.num_sketches):
      row = []
      for column_index in range(2 ** self.num_sketches):
        row.append(self.transition_probability(column_index, row_index))
      result.append(row)
    return numpy.array(result)

  def correction_matrix(self):
    """Matrix to denoise the vector of counts."""
    return numpy.linalg.inv(self.transition_matrix())

  def estimate_combinations(self):
    c = self.correction_matrix()
    v = self.sublegion_as_vector(0, self.sketch.m)
    return c.dot(v)

  def estimate_from_all(self):
    """Estimating cardinalities of Venn diagram from a noised Legion."""
    f = self.estimate_combinations()
    if self.num_sketches == 1:
      assert len(f) == 2
      return numpy.array(
          [0, self.sketch.get_cardinality_for_legionaries_count(f[1])])
    if self.num_sketches == 2:
      assert len(f) == 4
      a = self.sketch.get_cardinality_for_legionaries_count(f[1] + f[3])
      b = self.sketch.get_cardinality_for_legionaries_count(f[2] + f[3])
      a_or_b = self.sketch.get_cardinality_for_legionaries_count(
          f[1] + f[2] + f[3])
      # Ensuring that intersection will end up greater than 0 and smaller than
      # each of the components.
      # Note that due to correction matrix f[i] may be negative.
      a_or_b = max(min(a + b, a_or_b), a, b)
      return numpy.array(
          [0,
           a_or_b - b,
           a_or_b - a,
           a + b - a_or_b])
    else:
      # This can be generalized to an arbitrary set of sketches using
      # recursion.
      raise NotImplementedError


class Sampler(object):
  """Monte Carlo sampling posterior denoised sketches."""

  def __init__(self, sketch_list):
    self.number_to_combination_cache = {}
    self.num_sketches = len(sketch_list)
    assert 0 < self.num_sketches < 3, (
        'Only sampling of 1 or 2 sketches is implemented.')
    self.sketch_list = sketch_list
    self.flip_p = numpy.array([s.added_noise for s in self.sketch_list])
    self.stay_p = numpy.array([1 - s.added_noise for s in self.sketch_list])
    self.transition_matrix = self.get_transition_matrix()
    self.venn_estimator = VennEstimator(sketch_list)
    self.venn_cardinality_vector = self.venn_estimator.estimate_from_all()
    self.sketch = self.sketch_list[0].get_compatible_sketch()

  def transition_probability(self, a, b):
    """Probability of combination a flipped to combination b."""
    result = 1.0
    for i, (e_a, e_b) in enumerate(zip(a, b)):
      if e_a == e_b:
        result *= self.stay_p[i]
      else:
        result *= self.flip_p[i]
    return result

  def transition_probability_for_numbers(self, a, b):
    """Probability of combination number a go to combination number b."""
    combination_a = self.number_to_combination(a)
    combination_b = self.number_to_combination(b)
    return self.transition_probability(combination_a,
                                       combination_b)

  def number_to_combination(self, n):
    """Mapping combination number of a combination vector."""
    if n not in self.number_to_combination_cache:
      result = []
      i = n
      for _ in range(self.num_sketches):
        result.append(i % 2)
        i //= 2
      self.number_to_combination_cache[n] = numpy.array(result)
    return self.number_to_combination_cache[n]

  def get_transition_matrix(self):
    """Transition matrix between bit combinations.

    Returns:
      A square 2 ** self.num_sketches by 2 ** self.num_sketches matrix.
      Element at position result[i, j] is the probability of i-th combination to
      transition to j-th combination due to random flipping.
      Number i corresponds to a sequence of bits equal to the binary
      decomposition of i.
    """
    result = numpy.matrix(
        numpy.zeros((2 ** self.num_sketches,
                     2 ** self.num_sketches)))
    for i in range(2 ** self.num_sketches):
      for j in range(2 ** self.num_sketches):
        result[i, j] = self.transition_probability_for_numbers(i, j)
    return result

  def distribution_given_observation(self, observed_combination_number):
    """Probabilities of bit combinations given the obverved combination."""
    return numpy.array(
        self.transition_matrix[observed_combination_number, :]).reshape(
            (2 ** self.num_sketches,))

  def mask_probability_given_cadinality(self, index, venn_cardinality_vector):
    """Probability of each bit combination given Venn cardinality vector."""
    return self.sketch.hit_probability(index, venn_cardinality_vector)

  def diff_probability_given_cardinality(self, index, venn_cardinality_vector):
    """Probability of a bit set in a sketch representing set difference."""
    assert (self.sketch_list[0].added_noise == 0 and
            self.sketch_list[1].added_noise == 0), (
                'Probability can only be computed for clean sketches.')
    assert self.num_sketches == 2, (
        'Diff can only be computed for 2 sketches.')
    if self.sketch_list[0].sketch.get(index, 0) == 0:
      return 0.0
    if self.sketch_list[1].sketch.get(index, 0) == 0:
      return 1.0
    v = self.mask_probability_given_cadinality(
        index, venn_cardinality_vector)
    # Recall that
    #   v[1] is probability of A - B hitting the register,
    #   v[2] is probability of B - A hitting the register,
    #   v[3] is probability of A & B hitting the register.
    # Dividing possibilities that result in diff sketch bit set to 1 over
    # all possibilities that fit the observation.
    return (
        v[1] * v[2] * v[3] +
        v[1] * (1 - v[2]) * v[3] +
        v[1] * v[2] * (1 - v[3])) / (
            v[1] * v[2] * v[3] +
            v[1] * (1 - v[2]) * v[3] +
            v[1] * v[2] * (1 - v[3]) +
            (1 - v[1]) * v[2] * v[3] +
            (1 - v[1]) * (1 - v[2]) * v[3])

  def venn_probabilities_given_cardinality(
      self, index, venn_cardinality_vector):
    """Probability of each Venn component at index of sketch."""
    # Recall that
    #   v[1] is probability of A - B,
    #   v[2] is probability of B - A,
    #   v[3] is probability of A & B.
    v = self.mask_probability_given_cadinality(
        index, venn_cardinality_vector)
    if self.num_sketches == 1:
      return numpy.array([1 - v[1], v[1]])
    if self.num_sketches == 2:
      return numpy.array([
          (1 - v[1]) * (1 - v[2]) * (1 - v[3]),  # 0 0
          v[1] * (1 - v[2]) * (1 - v[3]),        # 1 0
          (1 - v[1]) * v[2] * (1 - v[3]),        # 0 1
          v[1] * v[2] * (1 - v[3]) + v[3]        # 1 1
      ])
    else:
      # This can be generalized using recursion.
      raise NotImplementedError

  def get_venn_priors(self, index):
    """Prior probabilities for Venn diagram at index."""
    return self.venn_probabilities_given_cardinality(
        index, self.venn_cardinality_vector)

  def get_all_venn_priors(self):
    """A matrix of priors of register combinations given cardinalities."""
    return numpy.array([self.get_venn_priors(i) for i in range(self.sketch.m)])

  def posterior_at_index(self, index):
    """Posterior distribution at index."""
    prop_to = self.get_venn_priors(index) * self.distribution_given_observation(
        self.venn_estimator.observation_at_index(index))
    return prop_to / sum(prop_to)

  def get_all_posteriors(self):
    """An array of aposteriori distributions of combinations of registers."""
    return numpy.array(
        [self.posterior_at_index(i) for i in range(self.sketch.m)])

  def sample_matrix(self):
    """A matrix with registers of sketches sampled from aposteriori."""
    posteriors = self.get_all_posteriors()
    combinations = [self.number_to_combination(i)
                    for i in range(2 ** self.num_sketches)]
    result = []
    combination_range = list(range(2 ** self.num_sketches))
    for posterior in posteriors:
      row = combinations[numpy.random.choice(combination_range, p=posterior)]
      result.append(row)
    return numpy.array(result)

  def sample(self):
    """A list of sketches sampled from aposteriori distribution."""
    m = self.sample_matrix()
    sketches = [self.sketch.get_compatible_sketch()
                for _ in range(self.num_sketches)]
    for register_index, row in enumerate(m):
      for sketch_index, e in enumerate(row):
        if e > 0:
          sketches[sketch_index].sketch[register_index] = e
    return sketches

  def sample_diff(self):
    """Samples an estimated sketch of set difference."""
    assert self.num_sketches == 2, 'Can only diff 2 sketches.'
    if (self.sketch_list[0].added_noise > 0 or
        self.sketch_list[1].added_noise > 0):
      pure_sketches = self.sample()
      pure_sampler = Sampler(pure_sketches)
    else:
      pure_sampler = self
    result = pure_sampler.sketch.get_compatible_sketch()
    for i in range(pure_sampler.sketch.m):
      p = pure_sampler.diff_probability_given_cardinality(
          i, pure_sampler.venn_cardinality_vector)
      bit = numpy.random.choice([0, 1],
                                p=[1 - p, p])
      if bit:
        result.sketch[i] = bit

    return result


class SequentialEstimator(base.EstimatorBase):
  """Estimator for LiquidLegions via sequential merge."""

  def __init__(self):
    pass

  @classmethod
  def sequential_merge(cls, sketch_list):
    """Returning result of sequential merge of the sketches."""
    assert sketch_list, 'I can only merge non-empty lists.'
    if len(sketch_list) == 1:
      sampler = Sampler(sketch_list)
      [result] = sampler.sample()
      return result

    result = sketch_list[0].get_compatible_sketch()
    sampler = Sampler(sketch_list[:2])
    clean_first, clean_second = sampler.sample()
    result.merge_in(clean_first)
    result.merge_in(clean_second)

    for sketch in sketch_list[2:]:
      sampler = Sampler([result, sketch])
      _, clean_sketch = sampler.sample()
      result.merge_in(clean_sketch)

    return result

  def __call__(self, sketch_list):
    """Estimating cardinality of the union."""
    if not sketch_list:
      return [0]  # That was easy!

    return [self.sequential_merge(sketch_list).get_cardinality()]
