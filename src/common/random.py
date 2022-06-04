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
"""Utilities for generating random numbers."""
import collections
import numpy as np

def choice_fast(n, m, random_state=np.random.RandomState()):
    """Chooses m numbers or objects from population n.

    O(m) space-optimal algorithm for generating m random indices for list 
    of size n without replacement. NumPy's built-in choice function is slow when
    replace=False because it allocates n-sized arrays. This method is around 
    1000x faster for the sizes of n and m we are dealing with.
    
    Args:
      n: list or integer to choose from. If n is a list, this method will return
        values in n. If n is an integer, this method will return indices.
      m: Number of elements to choose.
      random_state: RandomState object to control randomness.
    
    Returns:
      List of elements chosen from n or list of indices from 0 (inclusive) to n (exclusive)
    """
    assert isinstance(n, collections.abc.Iterable) or isinstance(n, int)
    # Get the maximum number as size
    if isinstance(n, int):
      size = n
    else:
      size = len(n)
    # We should always be choosing fewer than or up to size
    assert m <= size

    ### Robert Floyd's No-Replacement Sampling Algorithm ###
    ### https://fermatslibrary.com/s/a-sample-of-brilliance ###

    # Create an empty set to place numbers in
    s = set()
    s_add = s.add
    # Sample m random numbers and put them in the correct ranges:
    # First index should be sampled between 0 and size-m (inclusive)
    # Second index should be sampled between 0 and size-m+1 and so on
    # We cast to uint64 to floor the sampling.
    randints = (random_state.random_sample(m) 
               * (np.arange(size - m + 1, size + 1))).astype(np.uint64)
    for j in range(m):
        t = randints[j]
        if t in s:
            t = size - m + j
        s_add(t)
    assert len(s) == m

    # Turn set into numpy array. This is an array of randomly chosen indices
    # from 0 (inclusive) to n (exclusive)
    ret = np.fromiter(s, np.int64, m)
    # If the input was an int, return these indices
    if isinstance(n, int):
      return ret
    # Otherwise, return the elements of n at these indices
    return n[ret]
