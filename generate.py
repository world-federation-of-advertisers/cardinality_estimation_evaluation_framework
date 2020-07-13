import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import collections
import time
import random
import matplotlib.pyplot as plt
# import random
def generate_indices(n, m, random_state=np.random.RandomState()):
    """O(m) space-optimal algorithm for generating m random indices for list 
    of size n without replacement
    
    Args:
      n: list or integer to choose from. If n is a list, this method will return
      values. If n is an integer, this method will return indices.

      m: Number of elements to choose.

      random_state: RandomState object to control randomness.
    
    Returns:
      List of elements chosen from n or list of indices from 0 (inclusive) to n (exclusive)
    """
    # assert isinstance(n, collections.abc.Iterable) or isinstance(n, int)
    # Get the maximum number as size
    if isinstance(n, int):
      size = n
    else:
      size = len(n)
    # We should always be choosing fewer than the size
    # assert m <= size

    ### Robert Floyd's No-Replacement Sampling Algorithm ###
    # Create an empty set to place numbers in
    s = set()
    s_add = s.add
    # Sample m random numbers and put them in the correct ranges:
    # First index should be sampled between 0 and size-m (inclusive)
    # Second index should be sampled between 0 and size-m+1 and so on
    # We cast to uint64 to floor the sampling.
    randints = (random_state.random_sample(m) * (np.arange(size - m + 1, size + 1))).astype(np.uint64)
    for j in range(m):
        t = randints[j]
        if t in s:
            t = size - m + j
        s_add(t)
    # assert len(s) == m

    # Turn set into numpy array
    ret = np.fromiter(s, np.long, m)
    # If the input was an int, return the indices
    if isinstance(n, int):
      return ret
    # Otherwise, return the elements from the indices
    return n[ret]


def time_func(func, args, trials):
    past = time.perf_counter()
    for _ in range(trials):
        func(*args)
    now = time.perf_counter()
    return (now - past) / trials

n_pop = range(10**7, 5 * 10**7, int(0.5 * 10**7))
m_pop = [100000]
trials = 5
for m in m_pop:
    fast_times = []
    python_times = []
    slow_times = []
    print(f'Trials for {m:.1E}')
    for n in n_pop:
        print(n, end='\r', flush=True)
        fast_times.append(time_func(generate_indices, (n,m), trials))
        python_times.append(time_func(lambda n, m: np.array(random.sample(n, m)), (range(n), m), trials))
        slow_times.append(time_func(np.random.choice, (n,m, False), trials))
    print(f'Fast: {fast_times}')
    print(f'Python: {python_times}')
    print(f'Slow: {slow_times}')
    plt.clf()
    fast_line, python_line, slow_line = plt.plot(n_pop, fast_times, 'r-', 
                                              n_pop, python_times, 'g-', 
                                              n_pop, slow_times, 'b-')
    plt.legend([fast_line, python_line, slow_line], ["Robert Floyd's", "Naive Python", "np.random.choice"])
    plt.xlabel('Size of Population')

    plt.ylabel('Time (seconds)')
    plt.yscale('log')
    plt.grid(True)
    plt.title(f'Time Taken to Choose {m} Numbers')
    plt.savefig(f'plot_{n}_{m}.png')
       

