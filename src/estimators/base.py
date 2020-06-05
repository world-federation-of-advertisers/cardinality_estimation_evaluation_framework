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
"""Templates for cardinality estimators."""


class SketchBase(object):
  """Abstract class for sketches."""

  def add(self, x):
    """Add one ID to the sketch."""
    raise NotImplementedError()

  def add_ids(self, iterable):
    """Add multiple IDs to the sketch.

    Can be overridden in cases where there is a faster way to add multiple ids.

    Args:
      iterable: an iterable containing integer user_ids
    """
    for id_val in iterable:
      self.add(id_val)

  @staticmethod
  def get_sketch_factory():
    """Returns a function Handle which takes a np.random.RandomState as an arg.

    This function handle, when called, will return a fully-formed Sketch object,
    ready to generate sketches.
    """

    def f(random_state):
      _ = random_state
      raise NotImplementedError()

    _ = f
    # In an implementation, you would return f here
    # return f
    raise NotImplementedError()


class EstimatorBase(object):
  """An estimator takes a sketch and produces a cardinality estimate.

  Estimators exist apart from sketches because many types of sketches are
  amenable to having multiple types of estimators operate on them
  """

  def __call__(self, sketch_list):
    """Estimates the cardinality of the union of a list of sketches."""
    raise NotImplementedError()


class SketchNoiserBase(object):
  """A sketch noiser takes a sketch, copies it, and returns a noisy sketch.

  Similar to Estimators, Noisers exist apart from sketches because there
  are multiple ways to apply noise to a sketch.

  Note that while many Noisers may take in Differential Privacy parameters such
  as epsilon or delta, that we are making no guarantees that they are truly
  differentially private and suitable for protecting real user data.  The noise
  being added is for statistical accuracy purposes only, and does not include
  protections against certain attacks such as the 'Least Significant Digits'
  problem: https://crysp.uwaterloo.ca/courses/pet/F18/cache/Mironov.pdf

  A subclass of this class may chose to implement full protection, but it must
  explicitly specify that in its documentation.
  """

  def __call__(self, sketch):
    """Return a noised copy of the incoming sketch."""
    raise NotImplementedError()


class EstimateNoiserBase(object):
  """An estimate noiser adds noise to a cardinality estimate.

  Note that while many Noisers may take in Differential Privacy parameters such
  as epsilon or delta, that we are making no guarantees that they are truly
  differentially private and suitable for protecting real user data.  The noise
  being added is for statistical accuracy purposes only, and does not include
  protections against certain attacks such as the 'Least Significant Digits'
  problem: https://crysp.uwaterloo.ca/courses/pet/F18/cache/Mironov.pdf

  A subclass of this class may chose to implement full protection, but it must
  explicitly specify that in its documentation.
  """

  def __call__(self, cardinality_estimate):
    """Return a cardinality estimate with noise.

    Args:
      cardinality_estimate: A float, list or numpy vector of values to be
      noised.
    Returns:
      A noised float, list or numpy vector of the same type and dimenions as
      cardinality_estimate.
    """
    raise NotImplementedError()

