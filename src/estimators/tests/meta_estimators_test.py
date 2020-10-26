# Lint as: python3
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
"""Tests for meta_estimators.py."""
from absl.testing import absltest
from absl.testing import parameterized
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import ExponentialBloomFilter
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import FirstMomentEstimator
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import LogarithmicBloomFilter
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import UniformBloomFilter
from wfa_cardinality_estimation_evaluation_framework.estimators.meta_estimators import MetaVoCEstimator

test_params = [
    (UniformBloomFilter, {}, 'uniform'),
    (LogarithmicBloomFilter, {}, 'log'),
    (ExponentialBloomFilter, {'decay_rate': 10}, 'exp'),
]


class MetaVoCTest(parameterized.TestCase):

  @parameterized.parameters(*test_params)
  def test_single_insertion(self, bf, bf_kwargs, method):
    # Inserting only a single element into MetaEstimator should be
    # equivalent to FirstMomentEstimator for any number of buckets.
    for num_buckets in range(50, 100):
      true_estimator = FirstMomentEstimator(method=method)
      estimator = MetaVoCEstimator(num_buckets=num_buckets,
                                   adbf_estimator=true_estimator)
      factory = bf.get_sketch_factory(length=num_buckets, **bf_kwargs)

      b1 = factory(random_seed=0)
      b1.add(1)
      b2 = factory(random_seed=0)
      b2.add(1)

      self.assertAlmostEqual(estimator([b1, b2]), true_estimator([b1, b2]))

  @parameterized.parameters(*test_params)
  def test_incompatible_different_lengths(self, bf, bf_kwargs, method):
    estimator = MetaVoCEstimator(
        num_buckets=10,
        adbf_estimator=FirstMomentEstimator(method=method))

    factory1 = bf.get_sketch_factory(length=10, **bf_kwargs)
    factory2 = bf.get_sketch_factory(length=15, **bf_kwargs)

    b1 = factory1(random_seed=0)
    b2 = factory2(random_seed=0)
    with self.assertRaises(AssertionError):
      estimator([b1, b2])

  @parameterized.parameters(*test_params)
  def test_incompatible_different_seeds(self, bf, bf_kwargs, method):
    estimator = MetaVoCEstimator(
        num_buckets=10,
        adbf_estimator=FirstMomentEstimator(method=method))

    factory = bf.get_sketch_factory(length=10, **bf_kwargs)

    b1 = factory(random_seed=0)
    b2 = factory(random_seed=1)
    with self.assertRaises(AssertionError):
      estimator([b1, b2])


if __name__ == '__main__':
  absltest.main()
