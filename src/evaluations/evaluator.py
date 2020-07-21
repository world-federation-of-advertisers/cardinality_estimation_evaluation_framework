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
"""Evaluator for cardinality estimators."""

import copy
import itertools
import os
import pickle
import shutil
import time

from absl import logging
import numpy as np
from pathos.multiprocessing import ProcessPool, cpu_count
from tqdm import tqdm

from wfa_cardinality_estimation_evaluation_framework.evaluations.configs import EvaluationConfig
from wfa_cardinality_estimation_evaluation_framework.simulations.simulator import Simulator
from wfa_cardinality_estimation_evaluation_framework.simulations.simulator import SketchEstimatorConfig

# Pickle filenames for saving the configurations.
SCENARIO_SEED_FILE = 'set_generator_seed.p'
SCENARIO_CONFIG_FILE = 'scenario_config.p'
ESTIMATOR_CONFIG_FILE = 'sketch_estimator_config.p'
RUN_DIRS_FILE = 'run.p'
EVALUATION_RUN_TIME_FILE = 'evaluation_run_time'

# Keys of the dictionary to remember where the configs and results are saved.
KEY_OUT_DIR = 'output'
KEY_RUN_DIR = 'run'
KEY_EVALUATION_DIR = 'evaluation'
KEY_ESTIMATOR_DIRS = 'estimators'

# File names
RAW_RESULT_DF_FILENAME = 'df.csv'
AGG_RESULT_DF_FILENAME = 'df_agg.csv'


def _create_directory_tree(run_name, evaluation_config,
                           sketch_estimator_config_list, out_dir, overwrite):
  """Create directory for storing the results and configs.

  The directory will be:
  out_dir/
      /run_name/
          /evaluation_name/
              /sketch_estimator_config_list[0].name/
                  /evaluation_config.scenario_config_list[0].name
                  /evaluation_config.scenario_config_list[1].name
                  ...
              /sketch_estimator_config_list[1].name/
                  /evaluation_config.scenario_config_list[0].name
                  /evaluation_config.scenario_config_list[1].name
                  ...
              ...

  Args:
    run_name: a customized name of the evaluation run.
    evaluation_config: an EvaluationConfig object.
    sketch_estimator_config_list: a sequence of SketchEstimatorConfig, which
      will be evaluated under all the scenarios in evaluation_config.
    out_dir: an optional output directory.
    overwrite: a boolean variable. If set to True, will allow to overwrite the
      results even if the run exists. Otherwise, will raise error.

  Returns:
    A dictionary that contain all the directories of the evaluation, estimators
    and scenarios.
  """
  if out_dir is None:
    out_dir = os.getcwd()

  # Remove an existing directory tree if allow overwrite.
  if overwrite and os.path.exists(os.path.join(out_dir, run_name)):
    shutil.rmtree(os.path.join(out_dir, run_name))

  description_to_file_dir = {KEY_OUT_DIR: out_dir}
  description_to_file_dir[KEY_RUN_DIR] = os.path.join(
      description_to_file_dir[KEY_OUT_DIR], run_name)
  description_to_file_dir[KEY_EVALUATION_DIR] = os.path.join(
      description_to_file_dir[KEY_RUN_DIR], evaluation_config.name)

  description_to_file_dir[KEY_ESTIMATOR_DIRS] = {}
  for sketch_estimator_config in sketch_estimator_config_list:
    # Get the directory of the estimators.
    description_to_file_dir[KEY_ESTIMATOR_DIRS][
        sketch_estimator_config.name] = os.path.join(
            description_to_file_dir[KEY_EVALUATION_DIR],
            sketch_estimator_config.name)

    # Get the direcory of all scenarios under each estimator.
    estimator_dir = os.path.join(
        description_to_file_dir[KEY_EVALUATION_DIR],
        sketch_estimator_config.name)
    estimator_dir_dict = {}
    for scenario_config in evaluation_config.scenario_config_list:
      scenario_dir = os.path.join(estimator_dir, scenario_config.name)
      os.makedirs(scenario_dir)
      estimator_dir_dict[scenario_config.name] = scenario_dir
    description_to_file_dir[sketch_estimator_config.name] = (
        estimator_dir_dict)

  return description_to_file_dir


def load_directory_tree(out_dir, run_name, evaluation_name):
  """Load the evaluation directory structure.

  The directory will be:
  out_dir/
      /run_name/
          /evaluation_name/
              /sketch_estimator_configs[0].name/
                  /evaluation_config.scenario_configs[0].name
                  /evaluation_config.scenario_configs[1].name
                  ...
              /sketch_estimator_configs[1].name/
                  /evaluation_config.scenario_configs[0].name
                  /evaluation_config.scenario_configs[1].name
                  ...
              ...

  Args:
    out_dir: the output directory.
    run_name: the name of the evaluation run.
    evaluation_name: the name of the evaluation type.

  Returns:
    A dictionary that contain all the directories of the evaluation, estimators
    and scenarios.
  """
  description_to_file_dir = {KEY_OUT_DIR: out_dir}
  description_to_file_dir[KEY_RUN_DIR] = os.path.join(out_dir, run_name)
  description_to_file_dir[KEY_EVALUATION_DIR] = os.path.join(
      description_to_file_dir[KEY_RUN_DIR], evaluation_name)

  description_to_file_dir[KEY_ESTIMATOR_DIRS] = {}
  estimators = [
      f for f in os.listdir(description_to_file_dir[KEY_EVALUATION_DIR])
      if not os.path.isfile(
          os.path.join(description_to_file_dir[KEY_EVALUATION_DIR], f))]
  for estimator in estimators:
    # Get the directory of the estimators.
    estimator_dir = os.path.join(
        description_to_file_dir[KEY_EVALUATION_DIR], estimator)
    description_to_file_dir[KEY_ESTIMATOR_DIRS][estimator] = estimator_dir

    # Get the direcory of all scenarios under each estimator.
    estimator_dir_dict = {}
    scenarios = [
        f for f in os.listdir(estimator_dir)
        if not os.path.isfile(os.path.join(estimator_dir, f))]
    for scenario in scenarios:
      estimator_dir_dict[scenario] = os.path.join(estimator_dir, scenario)
    description_to_file_dir[estimator] = estimator_dir_dict

  return description_to_file_dir

def aggregate_and_write_times(times, estimator_to_file, pbar=None):
  """Aggregates time spent by each estimator and writes time to file.

  Args:
    times: list of (time (s), sketch estimator name, scenario name)
    estimator_to_file: dictionary mapping estimator name to file
    pbar: tqdm progress bar object. Can be None.
  
  Returns:
    None
  
  Side Effects:
    Writes to files given by estimator_to_file if estimator_name is in times
  """
  # Aggregate time spent on each scenario by estimator
  performance_stats = dict()

  # Sum up the time taken on each process, grouped by estimator. 
  # This should provide similar runtime numbers serially or
  # in parallel because it calculates CPU time, not wall time.
  for elapsed_time, sketch_estimator_name, scenario_name in times:
    time_file = os.path.join(
      estimator_to_file[
          sketch_estimator_name],
      EVALUATION_RUN_TIME_FILE)

    if time_file not in performance_stats.keys():
      performance_stats[time_file] = 0
    performance_stats[time_file] += elapsed_time

    # Update progress
    if pbar is not None:
      pbar.update()
      pbar.set_description(f'Finished {sketch_estimator_name}-{scenario_name}')

  # Write performance stats
  for time_file, elapsed_time in performance_stats.items():
    with open(time_file, 'w') as f:
      f.write(str(elapsed_time))

class Evaluator(object):
  """Run evaluations for cardinality estinators."""

  def __init__(self, evaluation_config, sketch_estimator_config_list, run_name,
               scenario_random_state=np.random.RandomState(), out_dir=None,
               overwrite=False, workers=0):
    """Construct an evaluator.

    Args:
      evaluation_config: an EvaluationConfig object.
      sketch_estimator_config_list: a sequence of SketchEstimatorConfig, which
        will be evaluated under all the scenarios in evaluation_config.
      run_name: a name of the run.
      scenario_random_state: a seed for generating the same simulation data
        for different estimators.
      out_dir: an optional output directory. If not specified, will use
        the current working directory.
      overwrite: a boolean variable. If set to True, will allow to overwrite the
        results even if the run exists. Otherwise, will raise error. By default,
        set to False.
      workers: (integer) number of processes to spawn. If this is set to 1, 
        the evaluations will run serially. If this is set to 0 or less, the 
        evaluations will run on as many processes as there are CPUs.

    Raises:
      AssertionError: if the evaluation_config is not an EvaluationConfig,
        or if any element of sketch_estimator_config_list is not a
        SketchEstimatorConfig.
    """
    assert isinstance(evaluation_config, EvaluationConfig), (
        'evaluation_config is not an EvaluationConfig object.')
    for sketch_estimator_config in sketch_estimator_config_list:
      assert isinstance(sketch_estimator_config, SketchEstimatorConfig), (
          'sketch_estimator_config_list does not contain all '
          'SketchEstimatorConfig objects')
    assert isinstance(workers, int), (
        'Number of workers must be an int.')
    if workers < 1:
      self.workers = cpu_count()
    else:
      self.workers = workers
      
    logging.info(f'Number of workers: {self.workers}')
    
    self.evaluation_config = evaluation_config
    self.sketch_estimator_config_list = sketch_estimator_config_list

    if out_dir is None:
      out_dir = os.getcwd()

    self.description_to_file_dir = _create_directory_tree(
        run_name=run_name,
        evaluation_config=evaluation_config,
        sketch_estimator_config_list=sketch_estimator_config_list,
        out_dir=out_dir,
        overwrite=overwrite)

    with open(os.path.join(out_dir, run_name, RUN_DIRS_FILE), 'wb') as f:
      pickle.dump(self.description_to_file_dir, f)

    # This is to make sure that different estimators use the same simulation
    # data under the same scenario.
    scenario_random_states = {}
    for scenario_config in evaluation_config.scenario_config_list:
      scenario_random_states[scenario_config.name] = np.random.RandomState(
          scenario_random_state.randint(2**32 - 1))
    self.scenario_random_states = scenario_random_states

  def __call__(self):
    self.evaluate_all()

  def evaluate_all(self):
    """Evaluate all estimators under all scenarios."""
    # Get all combinations of (scenario, estimator)
    work_items = itertools.product(
                  self.evaluation_config.scenario_config_list,
                  self.sketch_estimator_config_list
                 )
    total_length = len(self.evaluation_config.scenario_config_list) * \
                len(self.sketch_estimator_config_list)

    # Progress visualization
    with tqdm(total=total_length) as pbar:
      # Start scenarios; returns a list of (time, sketch name, scenario name)
      # tuples for each process spawned.
      with ProcessPool(self.workers) as pool:
        times = pool.uimap(self._run_one_scenario_process, work_items)
      
      # While the scenarios are running, save estimator configs
      for sketch_estimator_config in self.sketch_estimator_config_list:
        self.save_estimator(sketch_estimator_config)

      aggregate_and_write_times(times,
                                self.description_to_file_dir[KEY_ESTIMATOR_DIRS],
                                pbar)

  def evaluate_estimator(self, sketch_estimator_config):
    """Evaluate one estimator under all the scenarios."""
    logging.info('====Estimator: %s', sketch_estimator_config.name)

    # Save an example of the sketch_estimator_config.
    self.save_estimator(sketch_estimator_config)

    for scenario_config in self.evaluation_config.scenario_config_list:
      self.run_one_scenario(scenario_config, sketch_estimator_config)

  def save_estimator(self, sketch_estimator_config):
    estimator_dir = self.description_to_file_dir[KEY_ESTIMATOR_DIRS][
        sketch_estimator_config.name]
    estimator = sketch_estimator_config.sketch_factory(0)
    sketch_estimator_config_file = os.path.join(
        estimator_dir, ESTIMATOR_CONFIG_FILE)
    with open(sketch_estimator_config_file, 'wb') as f:
      pickle.dump(estimator, f)

  def run_one_scenario(self, scenario_config, sketch_estimator_config):
    """Run evaluation for an estimator under a scenario."""
    logging.info('Scenario: %s', scenario_config.name)

    scenario_dir = self.description_to_file_dir[
        sketch_estimator_config.name][scenario_config.name]
    # Save an example of the scenario_config.
    gen = scenario_config.set_generator_factory(np.random.RandomState())
    scenario_config_file = os.path.join(scenario_dir, SCENARIO_CONFIG_FILE)
    with open(scenario_config_file, 'wb') as f:
      pickle.dump(gen, f)

    # Run simulations.
    df_raw_file = os.path.join(scenario_dir, RAW_RESULT_DF_FILENAME)
    df_agg_file = os.path.join(scenario_dir, AGG_RESULT_DF_FILENAME)
    with open(df_raw_file, 'w') as f1, open(df_agg_file, 'w') as f2:
      sim = Simulator(
          num_runs=self.evaluation_config.num_runs,
          set_generator_factory=scenario_config.set_generator_factory,
          sketch_estimator_config=sketch_estimator_config,
          set_random_state=copy.deepcopy(
              self.scenario_random_states[scenario_config.name]),
          file_handle_raw=f1,
          file_handle_agg=f2)
      _ = sim()

  def _run_one_scenario_process(self, args):
    """
    Spawnable process to evaluate a sketch estimator on a certain scenario.

    Args:
      args: tuple of (scenario config, sketch estimator config)
    Returns:
      tuple: (elapsed time, estimator name, scenario name)
    """
    scenario_config, sketch_estimator_config = args

    start_time = time.time()
    # This is where the bulk of the work happens
    self.run_one_scenario(scenario_config, sketch_estimator_config)
    elapsed_time = time.time() - start_time
    return elapsed_time, sketch_estimator_config.name, scenario_config.name
