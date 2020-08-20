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
"""Simulation data generators for frequency evaluation framework.

We have implemented the following simulation data:
* Homogeneous user activities with a publisher
* Heterogeneous user frequency
* Publisher constant frequency.

See the following for further details:

https://github.com/world-federation-of-advertisers/cardinality_estimation_evaluation_framework/blob/master/doc/cardinality_and_frequency_estimation_evaluation_framework.md#frequency-simulation-scenarios
"""
from absl import logging
import numpy as np

from wfa_cardinality_estimation_evaluation_framework.common.random import choice_fast
from wfa_cardinality_estimation_evaluation_framework.simulations.set_generator_base import SetGeneratorBase
from wfa_cardinality_estimation_evaluation_framework.simulations.set_generator_base import _SetSizeGenerator


class HomogeneousPmfMultiSetGenerator(SetGeneratorBase):
  """Homogeneous multiset generator for arbitrary Pmf.

  This generator returns a collection of multisets.  Each multiset
  is drawn at random from a universe of specified size, so the overlap
  probabilities of two sets are independent.  Each item in a multiset
  is assigned a frequency determined by a probability mass function 
  that is passed to the constructor.  There is one PMF per set.
  """

  def __init__(self, universe_size, set_sizes, pmfs, random_state):
    """Create a homogeneous multiset generator for an arbitrary pmf.

    Args:
      universe_size: An integer value that specifies the size of the whole
        universe from which the ids will be sampled.
      set_sizes: An iterable containing the size of each set, aka, the number of
        reached IDs.
      pmfs: Iterable of list of float, representing the shifted probability
        mass functions of the frequency distributions associated to each of 
        thet sets.  pmf[i][j] is the probability of frequency j+1 occurring 
        in set i.
      random_state: a numpy random state.

    Raises: AssertionError when pmf is not valid.
    """
    self.set_sizes = list(set_sizes)
    self.pmf_list = list(pmfs)

    assert len(self.set_sizes) == len(self.pmf_list), (
      'Number of sets does not match number of pmfs')
    assert all(sum(p) == 1.0 for p in self.pmf_list), (
      'At least one PMF does not sum to 1.0')
    
    self.universe_size = universe_size
    self.random_state = random_state

  def __iter__(self):
    for set_size, pmf in zip(self.set_sizes, self.pmf_list):
      set_ids = choice_fast(self.universe_size, set_size, self.random_state)
      freq_per_id = self.random_state.choice(len(pmf), size=set_size, p=pmf) + 1
      multiset_ids = []
      for i, freq in zip(set_ids, freq_per_id):
        multiset_ids += [i] * freq
      self.random_state.shuffle(multiset_ids)
      yield multiset_ids
    return self

  
class HomogeneousMultiSetGenerator(HomogeneousPmfMultiSetGenerator):
  """Homogeneous multiset generator.

  This generator returns the multisets as described in section
  Frequency scenario 1: Homogeneous user activities within a publisher of this
  doc:
  https://github.com/world-federation-of-advertisers/cardinality_estimation_evaluation_framework/blob/master/doc/cardinality_and_frequency_estimation_evaluation_framework.md#frequency-scenario-1-homogeneous-user-activities-within-a-publisher

  Homogeneous means that all the reached ID's have the same frequency
  distribution.

  As a reached ID may appear one or more times in the returned set,
  to differentiate with a normal set whose elements can only appear once, we
  use the term multiset.

  The frequency distribution is defined as a shifted-Poisson distribution.
  I.e., freq ~ Poission(freq_rate) + 1.
  """

  @classmethod
  def get_generator_factory_with_num_and_size(cls, universe_size, num_sets,
                                              set_size, freq_rates,
                                              freq_cap):

    def f(random_state):
      return cls(universe_size, list(_SetSizeGenerator(num_sets, set_size)),
                 freq_rates, random_state, freq_cap)

    return f

  @classmethod
  def get_generator_factory_with_set_size_list(cls, universe_size,
                                               set_size_list, freq_rates,
                                               freq_cap):

    def f(random_state):
      return cls(universe_size, set_size_list, freq_rates, random_state,
                 freq_cap)

    return f

  def _truncated_poisson_pmf(self, mu, max_freq):
    """Probability mass function values of the truncated and shifted poisson

    The PMF of the poisson distribution with parameter mu is given by:

       f(k) = exp(-mu) * mu^k / k!

    The truncated poisson pmf has value f(k) for k < max_freq.  
    For k = max_freq, the value is equal to 1 - sum_{i=0}^{k-1} f(k).

    Args:
      mu:  float, rate parameter
      max_freq: int, truncation point

    Returns:
      A list of floats of length max_freq+1, representing the values of the
      truncating poisson PMF.
    """
    assert mu > 0, "Invalid rate parameter"
    assert max_freq > 0, "Invalid frequency parameter"
    k = np.arange(max_freq-1)
    log_k_factorial = np.array([0] + list(np.cumsum(np.log(k[1:]))))
    log_poisson = -mu + k * np.log(mu) - log_k_factorial
    poisson_pmf = list(np.exp(log_poisson))
    poisson_pmf.append(1.0 - sum(poisson_pmf))
    return poisson_pmf

  def __init__(self, universe_size, set_sizes, freq_rates, random_state,
               freq_cap=100):
    """Create a homogeneous multiset generator.

    Args:
      universe_size: An integer value that specifies the size of the whole
        universe from which the ids will be sampled.
      set_sizes: An iterable containing the size of each set, aka, the number of
        reached IDs.
      freq_rates: An iterable of the same size as set_sizes, specifying the
        non-negative freq_rate parameter of the shifted-Possion distribution.
      random_state: a numpy random state.
      freq_cap: A positive integer which represents the maximum number
        of times an ID can be seen in the returned set. 

    Raises: AssertionError when (1) set_size_list and freq_rate_list do not
      have equal length, (2) elements of freq_rate_list are not all
      non-negative, or (3) freq_cap is not None or positive.
    """

    set_size_list = list(set_sizes)
    freq_rate_list = list(freq_rates)
    
    assert len(set_size_list) == len(freq_rate_list), (
        'set_sizes and freq_rates do not have equal length.')
    assert all([freq_rate >= 0 for freq_rate in freq_rate_list]), (
        'Elements of freq_rate_list should be non-negative.')
    assert freq_cap > 0, 'freq_cap should be positive.'

    poisson_pmfs = []
    for mu in freq_rate_list:
      poisson_pmfs.append(self._truncated_poisson_pmf(mu, freq_cap-1))

    super().__init__(universe_size, set_size_list, poisson_pmfs, random_state)


class HeterogeneousMultiSetGenerator(SetGeneratorBase):
  """Heterogeneous multiset generator.

  This generator returns the multisets as described in section
  Frequency scenario 2: Heterogeneous user frequency of this doc:

  https://github.com/world-federation-of-advertisers/cardinality_estimation_evaluation_framework/blob/master/doc/cardinality_and_frequency_estimation_evaluation_framework.md#frequency-scenario-2-heterogeneous-user-frequency
  Heterogeneous means that the reached ID's have different frequency
  distribution.

  As a reached ID may appear one or more times in the returned set,
  to differentiate with a normal set whose elements can only appear once, we
  use the term multiset.

  The frequency distribution is defined as a shifted-Poisson distribution.
  I.e., freq ~ Poission(freq_rate) + 1.
  """

  @classmethod
  def get_generator_factory_with_num_and_size(cls, universe_size, num_sets,
                                              set_size, gamma_params,
                                              freq_cap):

    assert num_sets == len(gamma_params), (
        'num_sets not equal to len(gamma_params)')

    def f(random_state):
      return cls(universe_size, list(_SetSizeGenerator(num_sets, set_size)),
                 gamma_params, random_state, freq_cap)

    return f

  @classmethod
  def get_generator_factory_with_set_size_list(cls, universe_size,
                                               set_size_list, gamma_params,
                                               freq_cap):

    assert len(set_size_list) == len(gamma_params), (
        'set_size_list and gamma_params do not have equal length.')

    def f(random_state):
      return cls(universe_size, set_size_list, gamma_params, random_state,
                 freq_cap)

    return f

  def __init__(self, universe_size, set_sizes, gamma_params, random_state,
               freq_cap=None):
    """Create a heterogeneous multiset generator.

    Args:
      universe_size: An integer value that specifies the size of the whole
        universe from which the ids will be sampled.
      set_sizes: An iterable specifying the size of each set, aka, the number of
        reached IDs.
      gamma_params: An iterable of the same size as set_sizes, specifying the
        shape and rate parameters of the gamma distributions.
      random_state: a numpy random state.
      freq_cap: An optional positive integer which represents the maximum number
        of times an ID can be seen in the returned set. If not set, will not
        apply this capping.

    Raises: AssertionError when (1) set_sizes and gamma_params do not
      have equal length, (2) elements of gamma_params are not all
      non-negative, or (3) freq_cap is not None or positive.
    """
    self.set_sizes = list(set_sizes)
    self.gamma_params = list(gamma_params)

    assert len(self.set_sizes) == len(self.gamma_params), (
        'set_sizes and gamma_params do not have equal length.')
    assert all([params[0] > 0 for params in self.gamma_params]), (
        'Gamma shape parameters must be positive.')
    assert all([params[1] > 0 for params in self.gamma_params]), (
        'Gamma rate parameters must be positive.')
    assert freq_cap is None or freq_cap > 0, (
        'freq_cap should be None or positive.')

    self.universe_size = universe_size
    self.freq_cap = freq_cap
    self.random_state = random_state

  def __iter__(self):
    for set_size, gamma_params in zip(self.set_sizes, self.gamma_params):
      set_ids = choice_fast(self.universe_size, set_size, self.random_state)
      rate_parameters = self.random_state.gamma(shape=gamma_params[0],
                                                scale=gamma_params[1],
                                                size=set_size)
      frequencies = self.random_state.poisson(lam=rate_parameters,
                                              size=set_size) + 1
      if self.freq_cap:
        frequencies = np.minimum(frequencies, self.freq_cap)
      multiset_ids = []
      for i, freq in zip(set_ids, frequencies):
        multiset_ids += [i] * freq
      self.random_state.shuffle(multiset_ids)
      yield multiset_ids
    return self


class PublisherConstantFrequencySetGenerator(HomogeneousPmfMultiSetGenerator):
  """Frequency scenario 3: Publisher constant frequency

  This generator returns the multisets as described in section
  Frequency scenario 3: Publisher constant frequency
  doc:

  https://github.com/world-federation-of-advertisers/cardinality_estimation_evaluation_framework/blob/master/doc/cardinality_and_frequency_estimation_evaluation_framework.md#frequency-scenario-3-publisher-constant-frequency

  In the publisher constant frequency scenario, each publisher serves
  the same number of impressions to each user.
  """

  @classmethod
  def get_generator_factory_with_num_and_size(cls, universe_size, num_sets,
                                              set_size, frequency):

    def f(random_state):
      return cls(universe_size, list(_SetSizeGenerator(num_sets, set_size)),
                 frequency, random_state)

    return f

  @classmethod
  def get_generator_factory_with_set_size_list(cls, universe_size,
                                               set_size_list, frequency):


    def f(random_state):
      return cls(universe_size, set_size_list, frequency, random_state)

    return f

  def __init__(self, universe_size, set_sizes, frequency, random_state):
    """Create a publisher constant frequency set generator.

    Args:
      universe_size: An integer value that specifies the size of the whole
        universe from which the ids will be sampled.
      set_sizes: An iterable containing the size of each set, aka, the number of
        reached IDs.
      frequency: The frequency that will be assigned to each of the generated
        IDs.
      random_state: a numpy random state.

    Raises: AssertionError when (1) non-positive set_size is set_size_list, 
      (2) frequency is non-positive.
    """

    set_size_list = list(set_sizes)
    
    assert all([s > 0 for s in set_size_list]), (
        'Non-positive set size was found in set_size_list.')
    assert frequency > 0, 'Non-positive frequency given'

    pmf_list = [[0] * (frequency - 1) + [1]] * len(set_size_list)
    super().__init__(universe_size, set_size_list, pmf_list, random_state)
