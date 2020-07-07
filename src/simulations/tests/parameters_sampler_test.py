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
"""Tests for wfa_cardinality_estimation_evaluation_framework.simulations.parameters_sampler."""
from absl.testing import absltest

from wfa_cardinality_estimation_evaluation_framework.simulations.parameters_sampler import ParameterSampler
from wfa_cardinality_estimation_evaluation_framework.simulations.parameters_sampler import ParameterSamplerConfig


class ParameterSamplerTest(absltest.TestCase):

  def test_sampler(self):
    # Generate samplers.
    universe_size_gen = (i + 10 for i in range(2))
    sample_universe_size_func = lambda: next(universe_size_gen)
    set_size_list_gen = ([i + 1, i + 2] for i in range(2))
    sample_set_size_list_func = lambda: next(set_size_list_gen)
    parameter_sampler_config_list = [
        ParameterSamplerConfig('universe_size', sample_universe_size_func),
        ParameterSamplerConfig('set_size_list', sample_set_size_list_func)]
    s = ParameterSampler(parameter_sampler_config_list)

    # Sample the first time.
    result = s()
    expected = {'universe_size': 10, 'set_size_list': [1, 2]}
    self.assertEqual(
        result, expected, 'Sample parameters incorrectly 1st time.')

    # Sample again.
    result = s()
    expected = {'universe_size': 11, 'set_size_list': [2, 3]}
    self.assertEqual(
        result, expected, 'Sample parameters incorrectly 2nd time.')


if __name__ == '__main__':
  absltest.main()
