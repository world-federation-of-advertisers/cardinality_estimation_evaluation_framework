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
'''This file contains a method to run the run_evaluation script as __main__.

We are required to run the run_evaluation.py file as __main__ in order for 
abseil's built-in flags to work properly. However, the default behavior of 
Python entry-points does not assign __main__ to the specified module.
Therefore, in order to package run_evaluation.py into a global command line 
script properly, we must create some way of running run_evaluation.py as 
__main__. This approach is the most cross-platform and Pythonic way to
accomplish the above. For more information, see PEP 338:
  https://www.python.org/dev/peps/pep-0338/
'''
import runpy

def main():
  runpy.run_module(
    'wfa_cardinality_estimation_evaluation_framework'
      '.evaluations.run_evaluation',
    run_name='__main__',
    alter_sys=True
  )

if __name__ == '__main__':
  main()
