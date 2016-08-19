# Statistics for Humans
[![Build Status](https://travis-ci.org/PierreSelim/verysimplestats.svg?branch=master)](https://travis-ci.org/PierreSelim/verysimplestats)
[![Coverage Status](https://codecov.io/github/PierreSelim/verysimplestats/coverage.svg?branch=master)](https://codecov.io/github/PierreSelim/verysimplestats?branch=master)

## Install

From PyPi

```
pip install verysimplestats
```

From GitHub (with pip)

```
pip install git+https://github.com/PierreSelim/verysimplestats.git
```

## Purity
Scientific code, requires correctness. Functional programming guarantees part of
the correctness thanks to purity. The important part is not being able to
represents values that do not exist. We chose to raise `ValueError` when input
data do not permit computations (instead of using `None`)

## Tests
The doctest can be launched with:

```
nosetests --with-doctest --with-coverage --cover-package=verysimplestats
```

## Examples

Mean value

```python
>>> import verysimplestats as stats
>>> stats.mean([1, 2, 3, 4, 5])
3.0
```

Median value

```python
>>> import verysimplestats as stats
>>> stats.median([5, 2, 6, 4, 1, 3])
3.5
```

Linear regression

```python
>>> import verysimplestats as stats
>>> lm = stats.linear_regression([1, 2, 3], [1, 3, 4.5])
>>> lm
LinearRegression([1, 2, 3], [1, 3, 4.5])
>>> (lm.slope, lm.intercept, lm.rsquared)
(1.7499999999999984, -0.6666666666666634, 0.9932432432432422)
>>> [round(e, 4) for e in lm.residuals]
[-0.0833, 0.1667, -0.0833]
```

Variance is computed only for list of length greater or equal to 2

```python
>>> import verysimplestats as stats
>>> stats.variance([1])
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "verysimplestats\statistics.py", line 79, in variance
    raise ValueError(msg.format(x=x))
ValueError: Variance only exists for list with at least 2 elements [1]
```

## Supported functions

| Functions                                                                | Examples                                          |
|:-------------------------------------------------------------------------|:--------------------------------------------------|
| `mean(x: list) -> float`                                                 | `mean([1, 2, 3, 4, 5])`                           |
| `median(x: list) -> float`                                               | `median([5, 2, 6, 4, 1, 3])`                      |
| `variance(x: list) -> float`                                             | `variance([1, 2, 3, 4, 5])`                       |
| `standard_deviation(x: list) -> float`                                   | `standard_deviation([1, 2, 3, 4, 5])`             |
| `covariance(x: list, y: list) -> float`                                  | `covariance([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])`    |
| `correlation(x: list, y: list) -> float`                                 | `correlation([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])`   |
| `rsquared(x: list, y: list) -> float`                                    | `rsquared([1, 2, 3], [1, 3, 4.5])`                |
| `ordinary_least_square(x: list, y: list) -> (float, float)`              | `ordinary_least_square([1, 2, 3], [1, 3, 4.5])`   |
| `linear_forecast(slope: float, intercept: float, value: float) -> float` | `linear_forecast(2, -1, 3)`                       |
| `residuals(slope: float, intercept: float, x: float, y: float) -> float` | `residuals(1.75, -0.667, [1, 2, 3], [1, 3, 4.5])` |
| `linear_regression(x: list, y: list) -> LinearRegression`                | `linear_regression([1, 2, 3], [1, 3, 4.5])`       |

## License (MIT)
The MIT License (MIT)

Copyright (c) 2016 Pierre-Selim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE
