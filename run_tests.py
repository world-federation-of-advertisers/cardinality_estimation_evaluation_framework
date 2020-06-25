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
"""Searches for and runs all tests in this repository."""
from absl import logging
from absl.testing import absltest

import glob
import subprocess
import os
import inspect
from importlib import import_module

# Folders which can have tests
TEST_FOLDERS = ['./tests', './src', './evaluations']


def find_and_add_tests():
    """Finds all tests in TEST_FOLDERS directories and adds them to the global
    namespace for this module.

    Side Effects:
        Adds tests to global namespace so that absltest.main() can find
        and run them.
    """
    # Get all python files
    list_of_files = []
    for folder in TEST_FOLDERS:
        # Check subdirectories and first level directory for python files
        list_of_files.extend(glob.glob(folder + '/**/*.py', recursive=True))

    # Get all python files that are in tests/ folders
    test_files = glob.fnmatch.filter(list_of_files, '*tests/*.py')

    # Turn file names into module names ready for import
    # (For example: ./tests/test.py --> tests.test)
    test_modules = map(
        lambda x: '.'.join(os.path.normpath(x[:-3]).split(os.path.sep)),
        test_files)

    logging.info(f'Tests found: ')

    # Loop through all modules, import that module,
    # then link all test cases from that module to the global namespace
    for module_name in test_modules:
        # It's a bad idea to place test cases in __init__.py files.
        # This script does not encourage nor fulfill this behavior.
        # We do allow __init__ to be used as a package name, but this
        # is an extreme corner case.
        if module_name.endswith('__init__'):
            continue

        logging.info(f'\tIn module {module_name}:')

        # Import the module by its module name
        module = import_module(module_name)

        # Loop over all members of the imported module
        for name, value in inspect.getmembers(module):
            # If the member is both a class and is a subclass of absltest.TestCase,
            # we will link it to the global namespace.
            if inspect.isclass(value) and issubclass(value, absltest.TestCase):
                logging.info(f'\t\t{name}')
                # Modifying globals here allows absltest.main() to find the TestCase
                globals()[f'TEST_CASE_{module_name}_{name}'] = value


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    find_and_add_tests()
    absltest.main()