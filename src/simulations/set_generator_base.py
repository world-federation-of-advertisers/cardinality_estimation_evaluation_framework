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
"""Set Generator Base Classes"""

class _SetSizeGenerator(object):
  """Get set_size_list from set_size (fixed for each set) and num_sets."""

  def __init__(self, num_sets, set_size):
    self.num_sets = num_sets
    self.set_size = set_size

  def __iter__(self):
    for _ in range(self.num_sets):
      yield self.set_size


class SetGeneratorBase(object):
  """Base object for generating test sets."""

  def __next__(self):
    raise NotImplementedError()

  @classmethod
  def get_generator_factory_with_num_and_size(cls):
    """Returns a function Handle which takes a np.random.RandomState as an arg.

    This function handle, when called, will return a fully-formed SetGenerator
    object, ready to generate sets.
    """

    def f(random_state):
      _ = random_state
      raise NotImplementedError()

    _ = f
    # In an implementation, you would return f here
    # return f
    raise NotImplementedError()

  @classmethod
  def get_generator_factory_with_set_size_list(cls):
    """Returns a function Handle which takes a np.random.RandomState as an arg.

    This function handle, when called, will return a fully-formed SetGenerator
    object, ready to generate sets.
    """

    def f(random_state):
      _ = random_state
      raise NotImplementedError()

    _ = f
    # In an implementation, you would return f here
    # return f
    raise NotImplementedError()


