# Methods for Estimating the Union of Multiple Sets

## Overview

This repo includes code for

*   Sketches, which create an approximation of a set
*   Noisers, which add noise to those sketches
*   Estimators, which union together a series of sketches and then estimate the
    size of the combined Sketch
*   SetGenerators to create a series of randomly drawn sets with different kinds
    of relationships between the created sets
*   Simulator to combine all of the above, calculate error statistics, and
    compare possible methods.
*   Evaluators run an ensemble of simulators

### Quickstart
It is recommended to use a virtual environment with this project. If you already
have one, you can skip to the next step.
The quickest way to set up a virtual environment is by running:
```
python3 -m venv env
source env/bin/activate
```
To install the requirements as well as the evaluation framework, simply run:
```
pip install -r requirements.txt
python setup.py install
```
After these steps, the code and its dependencies will be installed as Python packages.


### Example

To start with and get a sense for how this code all works together, check out:
examples/basic_comparison.py

Which will run the same experiments across multiple different estimation methods.

See [this example](examples/notebooks/install_on_google_colab.ipynb) for how to install the framework on Google CoLab.

## Contributing

### Sketches/Estimators/Noisers

Located in estimators/

We anticipate most additions to this repo coming in the form of new kinds of
sketches and estimators, both of which are found in the estimators folder.

To get started, you should subclass either SketchBase in estimators/base.py or
AnySketch in estimators/any_sketch. AnySketch is an abstraction on top of
SketchBase which can make things quicker to develop for certain classes of
sketch, but the abstraction in SketchBase is much simpler, so feel free to start
wherever your appetite for learning a new abstraction takes you.

As the simplest example of everything one needs to implement to get started with
your own estimator, look at: estimators/exact_set.py. This has all the machinery
to implement the most basic sketch, which isn't really a sketch at all, but an
exact representation of a set using python's built-in set data structure.

For a simple example of the AnySketch abstraction, take a look at the classic
Bloom Filter implementation in estimators/bloom_filter.py. The AnySketch
abstraction is particularly empowering for bloom-filter-style approaches.

#### Differential Privacy in Noisers

Note that while many Noisers may take in Differential Privacy parameters such as
epsilon and delta, that we are making no guarantees that they are truly
differentially private and suitable for protecting real user data. The noise
being added is for statistical accuracy purposes only, and does not include
protections against certain attacks such as the 'Least Significant Digits'
problem: https://crysp.uwaterloo.ca/courses/pet/F18/cache/Mironov.pdf which is
only one of many potential differential privacy 'gotchas'

### SetGenerators/Simulator(s)

Located in simulations/

If you have an idea for a more realistic way to represent multiple groups of
users across multiple kinds of publishers, or perhaps another corner case, you
would start here with a new SetGenerator sub class.

We don't anticipate the need for a separate simulator, but still feel free to
create a new one or make the current one better.

### Bringing everything together

Once you have implemented anything new from the above section(s), please do add
it to the following and make sure it works with the existing machinery:

*   tests/interoperability_test.py
*   examples/basic_comparison.py
