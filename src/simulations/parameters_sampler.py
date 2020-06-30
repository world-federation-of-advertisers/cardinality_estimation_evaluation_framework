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
"""Sample simulation parameters."""

import collections

_ParameterSamplerConfig = collections.namedtuple('ParameterSamplerConfig',
                                                 ['name', 'sampler'])


class ParameterSamplerConfig(_ParameterSamplerConfig):
  """A subclass of namedtuple for parameter distribution configuration.

  The arguments to the named tuple are as follows:
    name: The parameter name. It should be one of the keyword argument of the
      set generator.
    sampler: A callable that will return a value of the named parameter. The
      values returned from the callable can be either deterministic or random.

  For example,
  universe_size_sampler = ParameterSamplerConfig(
      name='universe_size',
      sampler=lambda: np.random.randint(low=1e6, high=2e6, size=1)[0]
  )
  """
  pass


class ParameterSampler(object):
  """A sampler for sampling parameters.

  For example, the sampled parameters can be passed to the set generators via
  **kwargs.
  """

  def __init__(self, parameter_sampler_config_list):
    """Construct the samplers for set generator parameters.

    Args:
      parameter_sampler_config_list: An iterable of ParameterSamplerConfig.
    """
    self.parameter_sampler_config_list = parameter_sampler_config_list

  def __call__(self):
    return {
        config.name: config.sampler()
        for config in self.parameter_sampler_config_list
    }
