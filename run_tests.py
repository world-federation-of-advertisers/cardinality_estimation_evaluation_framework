# Testing
from absl.testing import absltest

# Use these to import all tests
import glob
import subprocess
import os
import inspect
from importlib import import_module

# Get a list of all files that have not been git-ignored
list_of_files = subprocess.check_output("git ls-files",
                                        shell=True).splitlines()
# Convert to string representation (default representation is bytes)
str_list_of_files = list(map(lambda x: x.decode('utf-8'), list_of_files))
# Get all python files that are in tests/ folders
test_files = glob.fnmatch.filter(str_list_of_files, "*tests/*.py")
# Turn file names into module names ready for import
test_modules = map(
    lambda x: '.'.join(os.path.normpath(x[:-3]).split(os.path.sep)),
    test_files)
for module_name in test_modules:
    module = import_module(module_name)
    for name, value in inspect.getmembers(module):
        if isinstance(value, absltest.TestCase):
            globals()["TEST_CASE_%s_%s" % (module_name, name)] = value

if __name__ == '__main__':
    absltest.main()