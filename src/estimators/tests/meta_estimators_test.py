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
import copy
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import ExponentialBloomFilter
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import FirstMomentEstimator
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import LogarithmicBloomFilter
from wfa_cardinality_estimation_evaluation_framework.estimators.bloom_filters import UniformBloomFilter
from wfa_cardinality_estimation_evaluation_framework.estimators.meta_estimators import MetaEstimatorBase
from wfa_cardinality_estimation_evaluation_framework.estimators.meta_estimators import MetaVectorOfCountsEstimator

test_params = [
    (UniformBloomFilter, {}, 'uniform'),
    (LogarithmicBloomFilter, {}, 'log'),
    (ExponentialBloomFilter, {'decay_rate': 10}, 'exp'),
]


def fake_voc_noiser(voc):
  """A fake VectorOfCounts noiser that add one ID to the first bucket."""
  noised_voc = copy.deepcopy(voc)
  noised_voc.stats[0] += 1
  return noised_voc


class MetaEstimatorBaseTest(parameterized.TestCase):

  @parameterized.parameters(
      (1, [1, 0]),
      (-1, [0, 0]),
      (3, [1, 1]),
      (0.4, [0, 0]),
  )
  def test_construct_fake_adbf(self, num_active_registers, expected):
    template_adbf = ExponentialBloomFilter(length=2, decay_rate=1)
    b = MetaEstimatorBase._construct_fake_adbf(num_active_registers,
                                               template_adbf)
    np.testing.assert_equal(b.sketch, np.array(expected))


class MetaVectorOfCountsEstimatorTest(parameterized.TestCase):

  @parameterized.parameters(*test_params)
  def test_single_insertion(self, bf, bf_kwargs, method):
    # Inserting only a single element into MetaEstimator should be
    # equivalent to FirstMomentEstimator for any number of buckets.
    for num_buckets in range(50, 100):
      true_estimator = FirstMomentEstimator(method=method)
      estimator = MetaVectorOfCountsEstimator(num_buckets=num_buckets,
                                              adbf_estimator=true_estimator)
      factory = bf.get_sketch_factory(length=num_buckets, **bf_kwargs)

      b1 = factory(random_seed=0)
      b1.add(1)
      b2 = factory(random_seed=0)
      b2.add(1)

      self.assertAlmostEqual(estimator([b1, b2]), true_estimator([b1, b2]))

  @parameterized.parameters(*test_params)
  def test_incompatible_different_lengths(self, bf, bf_kwargs, method):
    estimator = MetaVectorOfCountsEstimator(
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
    estimator = MetaVectorOfCountsEstimator(
        num_buckets=10,
        adbf_estimator=FirstMomentEstimator(method=method))

    factory = bf.get_sketch_factory(length=10, **bf_kwargs)

    b1 = factory(random_seed=0)
    b2 = factory(random_seed=1)
    with self.assertRaises(AssertionError):
      estimator([b1, b2])

  def test_meta_noiser(self):
    estimator_without_noiser = MetaVectorOfCountsEstimator(
        num_buckets=2,
        adbf_estimator=FirstMomentEstimator(method='exp'),
        meta_sketch_noiser=None)
    estimator_with_noiser = MetaVectorOfCountsEstimator(
        num_buckets=2,
        adbf_estimator=FirstMomentEstimator(method='exp'),
        meta_sketch_noiser=fake_voc_noiser)

    factory = ExponentialBloomFilter.get_sketch_factory(
        length=8, decay_rate=10)
    b1 = factory(random_seed=8)
    b1.add(1)
    b2 = factory(random_seed=8)
    b2.add(2)
    adbf_sketch_list = [b1, b2]

    unnoised_meta_voc_list = (
        estimator_without_noiser._transform_adbf_into_meta_sketches(
            adbf_sketch_list))
    noised_meta_voc_list = (
        estimator_with_noiser._transform_adbf_into_meta_sketches(
            adbf_sketch_list))

    self.assertLen(unnoised_meta_voc_list, 2,
                   msg='Length of unnoised meta voc list is not correct.')
    self.assertLen(noised_meta_voc_list, 2,
                   msg='Length of noised meta voc list is not correct.')

    for unnoised_meta_voc, noised_meta_voc in zip(unnoised_meta_voc_list,
                                                  noised_meta_voc_list):
      self.assertTrue(
          all(noised_meta_voc.stats - unnoised_meta_voc.stats
              == np.array([1, 0])),
          msg='Noiser does not add noise correctly.')


if __name__ == '__main__':
  absltest.main()
