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
"""Tests for wfa_cardinality_estimation_evaluation_framework.simulations.simulator."""

import io

from absl.testing import absltest
import numpy as np
import pandas as pd

from wfa_cardinality_estimation_evaluation_framework.estimators.base import EstimateNoiserBase
from wfa_cardinality_estimation_evaluation_framework.estimators.base import EstimatorBase
from wfa_cardinality_estimation_evaluation_framework.estimators.base import SketchBase
from wfa_cardinality_estimation_evaluation_framework.estimators.exact_set import AddRandomElementsNoiser
from wfa_cardinality_estimation_evaluation_framework.estimators.exact_set import ExactMultiSet
from wfa_cardinality_estimation_evaluation_framework.estimators.exact_set import LosslessEstimator
from wfa_cardinality_estimation_evaluation_framework.simulations import set_generator
from wfa_cardinality_estimation_evaluation_framework.simulations.simulator import Simulator
from wfa_cardinality_estimation_evaluation_framework.simulations.simulator import SketchEstimatorConfig


def get_simple_simulator(sketch_estimator_config=None):
  if not sketch_estimator_config:
    sketch_estimator_config = SketchEstimatorConfig(
        name='exact_set-lossless', sketch_factory=ExactMultiSet,
        estimator=LosslessEstimator())
  set_generator_factory = (
      set_generator.IndependentSetGenerator.
      get_generator_factory_with_num_and_size(
          universe_size=1, num_sets=1, set_size=1))

  return Simulator(
      num_runs=1,
      set_generator_factory=set_generator_factory,
      sketch_estimator_config=sketch_estimator_config,
      sketch_random_state=np.random.RandomState(1),
      set_random_state=np.random.RandomState(2))


class RandomSketchForTestRandomSeed(SketchBase):

  @classmethod
  def get_sketch_factory(cls):

    def f(random_seed):
      return cls(random_seed=random_seed)

    return f

  def __init__(self, random_seed):
    self.cardinality = random_seed

  def add_ids(self, ids):
    _ = ids


class EstimatorForTestRandomSeed(EstimatorBase):

  def __call__(self, sketch_list):
    return [sketch_list[-1].cardinality]


class FakeEstimateNoiser(EstimateNoiserBase):

  def __init__(self):
    self._calls = 0

  def __call__(self, cardinality_estimate):
    self._calls += 1
    return 10


class FakeSetGenerator(set_generator.SetGeneratorBase):
  """Generator for a fixed collection of sets."""

  @classmethod
  def get_generator_factory(cls, set_list):

    def f(random_state):
      return cls(set_list)

    return f

  def __init__(self, set_list):
    self.set_list = set_list

  def __iter__(self):
    for s in self.set_list:
      yield s
    return self
  

class SimulatorTest(absltest.TestCase):

  def test_simulator_run_one(self):
    sim = get_simple_simulator()
    data_frame = sim.run_one()
    self.assertLen(data_frame, 1)
    for pub in data_frame['num_sets']:
      self.assertEqual(pub, 1)

  def test_simulator_run_one_with_estimate_noiser(self):
    fake_estimate_noiser = FakeEstimateNoiser()
    sketch_estimator_config = SketchEstimatorConfig(
        name='exact_set-lossless',
        sketch_factory=ExactMultiSet, estimator=LosslessEstimator(),
        estimate_noiser=fake_estimate_noiser)
    sim = get_simple_simulator(sketch_estimator_config)
    data_frame = sim.run_one()
    self.assertLen(data_frame, 1)
    self.assertEqual(data_frame['estimated_cardinality_1'].iloc[0], 10)
    self.assertEqual(fake_estimate_noiser._calls, 1)

  def test_simulator_run_all_and_aggregate(self):
    sim = get_simple_simulator()
    data_frames = sim.run_all_and_aggregate()
    self.assertLen(data_frames, 2)
    for pub in data_frames[0]['num_sets']:
      self.assertEqual(pub, 1)

  def test_simulator_run_all_and_aggregate_with_noise(self):
    rs = np.random.RandomState(3)
    sketch_estimator_config = SketchEstimatorConfig(
        name='exact_set-lossless',
        sketch_factory=ExactMultiSet,
        estimator=LosslessEstimator(),
        sketch_noiser=AddRandomElementsNoiser(num_random_elements=3,
                                              random_state=rs))
    sim = get_simple_simulator(sketch_estimator_config)

    data_frames = sim.run_all_and_aggregate()
    self.assertLen(data_frames, 2)
    for pub in data_frames[0]['num_sets']:
      self.assertEqual(pub, 1)
    self.assertEqual(data_frames[0]['estimated_cardinality_1'][0], 4)
    self.assertEqual(data_frames[0]['true_cardinality_1'][0], 1)
    self.assertEqual(data_frames[0]['relative_error_1'][0], 3)

  def test_simulator_run_all_and_aggregate_multiple_runs(self):
    sketch_estimator_config = SketchEstimatorConfig(
        name='exact_set-lossless',
        sketch_factory=ExactMultiSet, estimator=LosslessEstimator())
    set_generator_factory = (
        set_generator.IndependentSetGenerator.
        get_generator_factory_with_num_and_size(
            universe_size=1, num_sets=1, set_size=1))

    sim = Simulator(
        num_runs=5,
        set_generator_factory=set_generator_factory,
        sketch_estimator_config=sketch_estimator_config)

    data_frames = sim.run_all_and_aggregate()
    self.assertLen(data_frames, 2)
    self.assertLen(data_frames[0], 5)
    for pub in data_frames[0]['num_sets']:
      self.assertEqual(pub, 1)

  def test_simulator_run_all_and_aggregate_write_file(self):
    sketch_estimator_config = SketchEstimatorConfig(
        name='exact_set-lossless',
        sketch_factory=ExactMultiSet, estimator=LosslessEstimator())
    set_generator_factory = (
        set_generator.IndependentSetGenerator.
        get_generator_factory_with_num_and_size(
            universe_size=1, num_sets=1, set_size=1))

    file_df = io.StringIO()
    file_df_agg = io.StringIO()
    sim = Simulator(
        num_runs=5,
        set_generator_factory=set_generator_factory,
        sketch_estimator_config=sketch_estimator_config,
        file_handle_raw=file_df,
        file_handle_agg=file_df_agg)
    df, df_agg = sim()

    # Test if the saved data frame is the same as the one returned from the
    # simulator.
    file_df.seek(0)
    df_from_csv = pd.read_csv(file_df)
    pd.testing.assert_frame_equal(df, df_from_csv)

    file_df_agg.seek(0)
    df_agg_from_csv = pd.read_csv(file_df_agg,
                                  header=[0, 1], index_col=0)
    pd.testing.assert_frame_equal(df_agg, df_agg_from_csv)

  def test_get_sketch_same_run_same_random_state(self):
    sketch_estimator_config = SketchEstimatorConfig(
        name='exact_set-lossless',
        sketch_factory=RandomSketchForTestRandomSeed,
        estimator=EstimatorForTestRandomSeed())
    set_generator_factory = (
        set_generator.IndependentSetGenerator.
        get_generator_factory_with_num_and_size(
            universe_size=1, num_sets=2, set_size=1))
    sim = Simulator(
        num_runs=1,
        set_generator_factory=set_generator_factory,
        sketch_estimator_config=sketch_estimator_config)
    df, _ = sim()
    self.assertEqual(
        df.loc[df['num_sets'] == 1, 'estimated_cardinality_1'].values,
        df.loc[df['num_sets'] == 2, 'estimated_cardinality_1'].values)

  def test_get_sketch_different_runs_different_random_state(self):
    sketch_estimator_config = SketchEstimatorConfig(
        name='random_sketch-estimator_for_test_random_seed',
        sketch_factory=RandomSketchForTestRandomSeed,
        estimator=EstimatorForTestRandomSeed())
    set_generator_factory = (
        set_generator.IndependentSetGenerator.
        get_generator_factory_with_num_and_size(
            universe_size=1, num_sets=1, set_size=1))
    sim = Simulator(
        num_runs=2,
        set_generator_factory=set_generator_factory,
        sketch_estimator_config=sketch_estimator_config)
    df, _ = sim()
    self.assertNotEqual(
        df.loc[df['run_index'] == 0, 'estimated_cardinality_1'].values,
        df.loc[df['run_index'] == 1, 'estimated_cardinality_1'].values)

  def test_extend_histogram(self):
    self.assertEqual(Simulator._extend_histogram(None, [], 1), [0])
    self.assertEqual(Simulator._extend_histogram(None, [3, 2, 1], 1), [3])
    self.assertEqual(Simulator._extend_histogram(None, [3, 2, 1], 2), [3, 2])
    self.assertEqual(Simulator._extend_histogram(None, [3, 2, 1], 3), [3, 2, 1])
    self.assertEqual(Simulator._extend_histogram(None, [3, 2, 1], 5), [3, 2, 1, 0, 0])

  def test_multiple_frequencies(self):
    sketch_estimator_config = SketchEstimatorConfig(
        name='exact-set-multiple-frequencies',
        sketch_factory=ExactMultiSet,
        estimator=LosslessEstimator(),
        max_frequency=3)
    set_generator_factory = (
        FakeSetGenerator.get_generator_factory(
          [[1, 1, 1, 2, 2, 3], [1, 1, 1, 3, 3, 4]]))
    sim = Simulator(
        num_runs=1,
        set_generator_factory=set_generator_factory,
        sketch_estimator_config=sketch_estimator_config)
    df, _ = sim()
    expected_columns = ['num_sets', 'estimated_cardinality_1', 'estimated_cardinality_2',
                        'estimated_cardinality_3', 'true_cardinality_1',
                        'true_cardinality_2', 'true_cardinality_3', 'run_index',
                        'relative_error_1', 'relative_error_2', 'relative_error_3']
    expected_data = [
      [1, 3, 2, 1, 3, 2, 1, 0, 0., 0., 0.],
      [2, 4, 3, 2, 4, 3, 2, 0, 0., 0., 0.]
    ]
    
    expected_df = pd.DataFrame(expected_data, columns=expected_columns)
    pd.testing.assert_frame_equal(df, expected_df)

    
if __name__ == '__main__':
  absltest.main()
