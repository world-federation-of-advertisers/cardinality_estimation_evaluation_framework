# **Speeding Up Choice Operations**


### By pfhgetty@google.com
Reviewers: knightbrook@google.com

## Overview

In this repo, various data structures are tested to their limits in various simulation scenarios designed to give estimates on the accuracy of these data structures.

Simplistically, these simulation scenarios involve randomly choosing tens of thousands of random numbers from _0_ (inclusive) to _n_ (exclusive) without replacement, storing these random numbers in both a ground-truth (100% accurate) data structure and a probabilistic accurate data structure, and then evaluating the performance of the probabilistic data structure against the ground-truth. Each step of this process is not extremely complicated, but there is **large room for improvement in the step of choosing random numbers from _0_ to _n_ without replacement**, which is outlined in this document.

In the end, we produce a nearly **60x speedup **in some simulations, greatly increasing the speed at which we can iterate on designs and saving lots of memory in the process.


## <span>Choosing Random Numbers</span>

**The following sections will cover design and methodology. To see the results, skip to <a href='#results'>Results and Conclusion</a>.


### What’s in a Set?

Choosing random numbers is pretty straightforward. When choosing _m_ numbers from _0_ to _n_, a very simple implementation where _randint(a, b)_ samples random numbers from _a_ (inclusive) to _b_ (exclusive) would be as follows:


```
numbers = []
for i in range(m):
    numbers.append(randint(0, n))
```


This runs in _O(m)_ time, which is good (because _n_ is usually much larger), but doesn’t fully capture our requirements. We need the numbers to be sampled without replacement.

The following is a simple implementation of sampling without replacement that we can refer to as _naive Python_:


```
numbers = set()
while len(numbers) < m:
    numbers.add(randint(0, n))
```


This is a good solution that will run in _O(m)_ time as long as _m_ <&lt; _n_.
<a href='#bigo'>Of course, as m becomes closer to n, there will be more collisions of randint(0, n) with what we have already put in the set.</a>


### Comparing to NumPy

How does _np.random.choice_ compare against the naive Python solution? Well it turns out that the naive Python solution is 
already written in Python’s _random_ module so it’s quite easy to compare the naive Python to the NumPy version. 
When the population _n_ is 1000 (quite small) and _m_ is 4 (also quite small), the naive Python version is **nearly 25x faster**.
And unlike other NumPy methods where NumPy ends up surpassing Python as _n_ gets larger, in this case the NumPy method actually becomes significantly
slower than the Python version as _n_ gets larger, getting up to **nearly** **35x slower** (before the NumPy method actually runs into a memory error 
on my testing machine).

How could this be? Operations in C should be at least as efficient as Python operations if not much more performant. 
Well, it  turns out that 
[when taking a look at the code](https://github.com/numpy/numpy/blob/maintenance/1.16.x/numpy/random/mtrand/mtrand.pyx#L1032), 
NumPy’s _choice_ operation really boils down to:


```
def choice(n, m, replace):
    if replace:
        return np.random.permutation(n)[:m]
    else:
        # do some normal random sampling
```


This means that when we call the _choice_ function, NumPy is creating an _n_-sized array, randomly shuffling it, 
and then returning the first _m_ elements from this array. This is, at best, an _O(n)_ operation that also uses _O(n)_
memory which is much worse than our _O(m)_ operation that we wrote in the naive Python algorithm.


### Can We Do Better?

So now that we’ve identified a simple, naive Python solution that gives large increases in performance, what are the drawbacks of such a solution?

Well, the first is that the runtime is non-deterministic. Depending on the amount of _collisions_ 
or number of times we sample a number from 0 to _n_ that we have already sampled, the runtime of this method
could double or even triple without explanation, especially with large _m_.

The second is that the code is not easily vectorizable. Because of the collisions, the number of random numbers 
we need to sample is not known until we’ve sampled more than we need. Therefore, it is difficult to decide how many numbers we need
to sample beforehand in order to effectively vectorize the code.

So multiple problems are caused by the amount of numbers we randomly sample not being deterministic. 
Luckily, this problem has already been solved for us in 
[Robert Floyd’s algorithm](https://fermatslibrary.com/s/a-sample-of-brilliance) which goes:


```
s = set()
for j in range(n-m, n):
    t = randint(0, j+1)
    if t not in s:
        s.add(t)
    else
        s.add(j)
```


This elegant algorithm allows us to uniformly sample numbers from 0 to _n_ without replacement in _O(m)_ time (importantly, 
no matter the _m_ or _n_; _m_ <&lt; _n_ is not required)
with a deterministic number of _randint_ calls. 
[Various proofs of correctness have been outlined](https://math.stackexchange.com/questions/178690/whats-the-proof-of-correctness-for-robert-floyds-algorithm-for-selecting-a-sin), 
but the simplest explanation is that:


```
At the end of the first loop: 
The numbers from 0 to n-m each have had one chance to be chosen. 
At the end of the second loop: 
The numbers from 0 to n-m each have had two chances to be chosen.
The number n-m+1 has had two chances to be chosen.
At the end of the third loop:
	The numbers from 0 to n-m+1 each have had three chances to be chosen.
	The number n-m+2 has had three chances to be chosen.
And so on...
```


Floyd’s algorithm being deterministic allows us to vectorize the 
_randint_ call as a NumPy operation. We can accomplish this by creating an array of _m_ random floats from [0, 1):


```
randnums = np.random.random_sample(m)
```


Then scaling them such that the first index is scaled by _n - m + 1_, the second index is scaled by _n - m + 2_, and so on. 
We then cast the whole array to an int, flooring its values.


```
randints = (randnums * (np.arange(size-m+1, size+1))).astype(np.uint64)
```


We have now pre-sampled all of our random integers. Therefore, the rest is simply:


```
for j in range(m):
    t = randints[j]
    if t in s:
        t = size - m + j
    s.add(t)
```


This vectorization of choosing random numbers gives an almost **4x speedup** to the algorithm when _m_ is greater than 10,000 compared to choosing the
random integers within the loop.


## <span id='results'>Results and Conclusion</span>

We have now reached an efficient, deterministic, and vectorized form of our algorithm. We can now compare results with NumPy’s implementation.

![Graph should be here.](images/choice_comparison.png "Comparison of choice methods.")

Note the log scale on the y-axis. Using Robert Floyd’s method of random sampling, we get a **nearly 120x speedup **in performance of choosing random numbers which only increases as the population size (_n_) gets larger (the largest test I was able to do produced speedups of **nearly 960x **with _n > 10^9_). When run in the simulation, this translates into a **nearly 60x speedup **for running the whole simulation (using Vector of Counts and a population size of 10^9).


## <span>Other Notes</span>


### <span id='bigo'>Big-O of a Non-Deterministic Process</span>

We estimated _O(m)_ for the naive Python process, but in truth, that is only when m &lt;< n. 
To calculate the expected Big-O of the naive Python process, we can use the number of times we expect to sample for each intermediary length of our set 
(which will have a length of _m_ at the end of the process).


```
First iteration: 1 expected sample
For the first loop we expect to only have to sample once 
(there is nothing in our set to collide with)

Second iteration: n/(n-1) expected samples
For the second loop, we sample numbers from 0 to n and we may 
collide with 1 other number that we have already sampled. 

The probability of the collision is 1/n. Using a geometric distribution, 
we can expect to have to sample at least n/(n-1) times.
Third iteration: n/(n-2) expected samples

And so on…
```


This means our average Big-O will be the result of the sum:

<p align="center">
<img align="center" src="https://render.githubusercontent.com/render/math?math=\sum_{i=0}^{m-1}%20\frac{n}{n-1}%20=%20n%20\sum_{i=0}^{m-1}%20\frac{1}{n-1}%20=%20n*(H_n-H_{n-m})">
</p>

Where H<sub>n</sub> is the n<sup>th</sup> harmonic number of the harmonic series. The harmonic series can be approximated as 
<img src="https://render.githubusercontent.com/render/math?math=H_n=ln(n)%2B\gamma"> where <img src="https://render.githubusercontent.com/render/math?math=\gamma">
is the [Euler-Mascheroni constant](https://en.wikipedia.org/wiki/Euler%E2%80%93Mascheroni_constant). Therefore, the average Big-O then becomes:

<p align="center">
<img align="center" src="https://render.githubusercontent.com/render/math?math=n*(ln(n)%2B%5Cgamma-ln(n-m)-%5Cgamma)%3Dn*(ln(%5Cfrac%7Bn%7D%7Bn-m%7D))">
</p>

