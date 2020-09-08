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
"""Contains the class AnySketch and the objects required to configure it."""

from wfa_cardinality_estimation_evaluation_framework.estimators import any_sketch


class ExponentialSameKeyAggregator():
  """Implement a Same Key Aggregator in Exponential bloom filter."""

  @classmethod
  def get_sketch_factory(cls, length, decay_rate):

    def f(random_seed):
      return cls(length, decay_rate, random_seed)

    return f

  def __init__(self, length, decay_rate, random_seed):
    """Creates an ExponentialBloomFilter.

    Args:
       length: The length of bit vector for the Exponential bloom filter.
       decay_rate: The decay rate of Exponential distribution.
       random_seed: An optional integer specifying the random seed for
         generating the random seeds for hash functions.
    """
    self.unique_key_sketch = any_sketch.AnySketch.__init__(
        self,
        any_sketch.SketchConfig([
            any_sketch.IndexSpecification(
                any_sketch.ExponentialDistribution(length, decay_rate), "exp")
        ], num_hashes=1, value_functions=[any_sketch.UniqueKeyFunction()]),
        random_seed)
    self.frequency_count_sketch = any_sketch.AnySketch.__init__(
        self,
        any_sketch.SketchConfig([
            any_sketch.IndexSpecification(
                any_sketch.ExponentialDistribution(length, decay_rate), "exp")
        ], num_hashes=1, value_functions=[any_sketch.SumFunction()]),
        random_seed)

  def add(self, x):
    self.unique_key_sketch.add(x)
    self.frequency_count_sketch.add(x)
