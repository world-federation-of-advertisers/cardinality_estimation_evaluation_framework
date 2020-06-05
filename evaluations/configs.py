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
"""Evaluation framework configurations."""
import collections

_SketchEstimatorConfig = collections.namedtuple(
    'SketchEstimatorConfig',
    ['name', 'sketch_factory', 'estimator', 'noiser'])


# This class exists as a placeholder for a docstring.
class SketchEstimatorConfig(_SketchEstimatorConfig):
  """A subclass of namedtuple for providing a estimator config to the simulator.

  The arguments to the named tuple are as follows:
    name: A string representing the name of the estimator.
    sketch_factory: A callable that takes as a single argument a
      numpy.random.RandomState and returns a class that conforms to
      cardinality_estimator_base.Sketch.
    estimator: A class that conforms to cardinality_estimator_base.Estimator.
    noiser: A class that conforms to cardinality_estimator_base.Noiser.
  """
  pass


_ScenarioConfig = collections.namedtuple(
    'ScenarioConfig', ['name', 'set_generator_factory'])


# This class exists as a placeholder for a docstring.
class ScenarioConfig(_ScenarioConfig):
  """A namedtuple subclass providing a scenario config to the EvaluationConfig.

  The arguments to the named tuple are as follows:
    name: the name of the scenario.
    set_generator_factory: A callable that takes as a single argument a
      numpy.random.RandomState and returns a class that conforms to
      set_generator.SetGeneratorBase.
  """
  pass


_EvaluationConfig = collections.namedtuple(
    'EvaluationConfig', ['name', 'num_runs', 'scenario_config_list'])


# For documenting the arguments.
class EvaluationConfig(_EvaluationConfig):
  """A namedtuple subclass of the evaluation configurations.

  It needs the following arguments:
    name: the name of the evaluation config.
    num_runs: the number of runs for each scenario.
    scenario_config_list: a sequence of ScenarioConfig.
  """
  pass
