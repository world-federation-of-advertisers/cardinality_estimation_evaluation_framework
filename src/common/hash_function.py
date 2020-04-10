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
"""Helper functions around various hash functions possibilities."""

import math
import farmhash

MAX_HASH_VALUE = 2**64-1


class HashFunction(object):
  """A wrapper around 64bit farmhash that supports equality testing and sorting.

  This class exists so that we can ensure comptibility of sketches when
  an estimator is combining them.
  """

  def __init__(self, random_seed, modulus=None):
    """Initialize a hash function.

    Args:
      random_seed: The random seed to use when hashing inputs
      modulus: The value with which to mod the hash value. If None just
        return the raw hash value.

    """
    self._random_seed = int(random_seed)
    if modulus is not None and int(math.log2(modulus)) > 64:
      raise ValueError('This hash function only outputs 64 bits max')
    self._modulus = modulus

  @property
  def random_seed(self):
    """Gets the random seed."""
    return self._random_seed

  @property
  def modulus(self):
    """Gets the modulus."""
    return self._modulus

  def __call__(self, x):
    """Returns the hash x modulus self.modulus."""
    val = farmhash.hash64withseed(str(x), self._random_seed)
    return val if self.modulus is None else val % self._modulus

  def __eq__(self, other):
    """Returns true if the HashFunctions have the same seed and modulus."""
    return (isinstance(other, HashFunction) and
            self.random_seed == other.random_seed and
            self.modulus == other.modulus)

  def __lt__(self, rhs):
    """Returns true if self is to be considered less than rhs."""
    assert isinstance(rhs, HashFunction), 'expected rhs to be a HashFunction'
    return self.modulus < rhs.modulus or self.random_seed < rhs.random_seed
