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
from absl import app
from absl import flags
from absl import logging
from absl.testing import absltest

import glob
import os
import inspect
from importlib import import_module
from functools import reduce

FLAGS = flags.FLAGS

flags.DEFINE_list(
    'folders', ['./'], 'Comma-separated list of folders to run all tests in. \
        If no input is given, runs all tests in this directory.')
# Make sure all folders specified exist.
flags.register_validator(
    'folders',
    lambda folder: reduce(lambda x, y: x and y, map(os.path.exists, folder)),
    message="Specified folder(s) does not exist.")


def find_modules(folders):
    """Returns a list of all modules in test_folders directories.

    Args:
      folders: list of folders to check for modules
    
    Returns:
      A list of tuples (module_name, imported module)
    """
    # Get all python files
    list_of_files = []
    for folder in folders:
        # Check subdirectories and first level directory for python files
        list_of_files.extend(glob.glob(folder + '/**/*.py', recursive=True))

    # Get all python files that are in tests/ folders
    test_files = glob.fnmatch.filter(list_of_files, '*tests/*.py')

    # Turn file names into module names ready for import
    # (For example: ./tests/test.py --> tests.test)
    def file_name_to_module_name(filename):
        # Remove .py extension
        rem_py = filename[:-3]
        # Normalize filepath (./tests\\test//smth --> ./tests/test/smth)
        normed = os.path.normpath(rem_py)
        # Split on normalized path separator
        split = normed.split(os.path.sep)
        # Add periods to create module name
        return '.'.join(split)

    test_modules = map(file_name_to_module_name, test_files)
    # Filter out python modules from virtual environments
    filter_func = lambda module_name: "lib.python" not in module_name \
                                  and "lib64.python" not in module_name
    test_modules = filter(filter_func, test_modules)


    modules = []
    # Import modules and filter out ones that cannot be imported.
    for module_name in test_modules:
        module = import_module(module_name)
        modules.append((module_name, module))

    return modules


def add_tests(modules):
    """From a list of modules, adds all TestCases to the global namespace
    such that absltest.main() finds and runs all TestCases from modules.

    Args:
      modules: list of (module_name, imported module)
    
    Returns:
      None
    
    Side Effects:
        Adds tests to global namespace so that absltest.main() can find
        and run them.
    """
    logging.info(f'Tests added: ')
    # Loop through all modules, import that module,
    # then link all test cases from that module to the global namespace
    for module_name, module in modules:
        # It's a bad idea to place test cases in __init__.py files.
        # This script does not encourage nor fulfill this behavior.
        # We do allow __init__ to be used as a package name, but this
        # is an extreme edge case.
        if module_name.endswith('__init__'):
            continue

        logging.info(f'  In module {module_name}:')

        # Loop over all members of the imported module
        for name, value in inspect.getmembers(module):
            # If the member is both a class and is a subclass of absltest.TestCase,
            # we will link it to the global namespace.
            if inspect.isclass(value) and issubclass(value, absltest.TestCase):
                logging.info(f'\t{name}')
                # Modifying globals here allows absltest.main() to find the TestCase
                globals()[f'TEST_CASE_{module_name}_{name}'] = value


def find_and_add_tests(folders):
    """Finds all modules in list of folders and adds all TestCases
    to the global namespace.
    
    Args:
      folders: list of folders to check for modules with TestCases
    """
    modules = find_modules(folders)
    add_tests(modules)


def main(argv):
    logging.set_verbosity(logging.INFO)
    find_and_add_tests(FLAGS.folders)
    absltest.main()


if __name__ == '__main__':
    app.run(main)
