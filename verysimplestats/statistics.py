"""Statistics for humans

I lacked a very simple lib for statistics so here it is. It provides really
simplistic float computation statistics indicators.
"""


import math


def mean(x):
    """Mean.

    Args:
        data (list): float (int) list

    Examples:
        >>> mean([1, 2, 3, 4, 5])
        3.0

        >>> mean([])
        Traceback (most recent call last):
        ...
        ValueError: An empty data list has no mean []

        >>> mean(42)
        Traceback (most recent call last):
        ...
        TypeError: 'int' object is not iterable
    """
    if not x:
        raise ValueError("An empty data list has no mean {x}".format(x=x))
    return math.fsum(x) / len(x)


def median(x):
    """Median.

    When x has an even number of element, mean of the two central value of the
    distribution.


    Examples:
        >>> median([5, 2, 4, 1, 3])
        3.0

        >>> median([5, 2, 6, 4, 1, 3])
        3.5
    """
    y = sorted(x)
    middle = len(x) / 2
    if len(x) % 2 == 0:
        return (y[int(middle - 0.5)] + y[int(middle + 0.5)]) / 2.0
    else:
        return float(y[middle])


def variance(x):
    """Unbiased sample variance.

    The population variance is (n-1)/n * variance(x)

    Args:
        x (list): float (int) list

    Examples:
        >>> variance([1, 2, 3, 4, 5])
        2.5

        >>> variance([42, 42, 42])
        0.0

        >>> variance([1])
        Traceback (most recent call last):
        ...
        ValueError: Variance only exists for list with at least 2 elements [1]
    """
    if len(x) < 2:
        msg = "Variance only exists for list with at least 2 elements {x}"
        raise ValueError(msg.format(x=x))
    m = mean(x)
    return math.fsum([math.pow(xi - m, 2) for xi in x]) / (len(x) - 1)


def standard_deviation(x):
    """Unbiased sample standard deviation.

    Args:
        x (list): float (int) list

    Examples:
        >>> round(standard_deviation([1, 2, 3, 4, 5]), 13)
        1.5811388300842
    """
    return math.sqrt(variance(x))


def covariance(x, y):
    """Unbiased sample covariance.

    Examples:
        >>> covariance([1, 2], [0, 0])
        0.0

        >>> covariance([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
        2.5

        >>> covariance([1, 2], [-1, -2, -3])
        Traceback (most recent call last):
        ...
        ValueError: Covariance is defined for vector of same length
    """
    if len(x) != len(y):
        msg = "Covariance is defined for vector of same length"
        raise ValueError(msg)
    xy = [xi * y[i] for i, xi in enumerate(x)]
    return (len(x) / (len(x) - 1.0)) * (mean(xy) - (mean(x) * mean(y)))


def correlation(x, y):
    """Pearson's correlation coefficient.

    Args:
        x (list): float or int list
        y (list): float or int list

    Examples:
        >>> round(correlation([1, 2, 3], [-1, -2, -3]), 13)
        -1.0

        >> correlation([-2, -1, 0, 1, 2], [2, 1, 0, 1, 2])
        0.0

        >>> round(correlation([1, 2, 3, 4, 5], [-1, -2, -3, -3, -5]), 7)
        -0.9594032

        >>> correlation([1, 2], [-1, -2, -3])
        Traceback (most recent call last):
        ...
        ValueError: Correlation is defined for vector of same length
    """
    if len(x) != len(y):
        msg = "Correlation is defined for vector of same length"
        raise ValueError(msg)
    return covariance(x, y) / (standard_deviation(x) * standard_deviation(y))


def rsquared(x, y):
    """R squared.

    Examples:
        >>> round(rsquared([1, 2, 3], [-1, -2, -3]), 13)
        1.0
    """
    return math.pow(correlation(x, y), 2)


def linear_regression(x, y):
    """Linear regression based on least sum of square.

    Args:
        x (list): float or int list
        y (list): float or int list

    Examples:
        >>> fit = linear_regression([1, 2, 3, 4, 5], [4, 4.5, 5.5, 5.3, 6])
        >>> round(fit['slope'], 13), round(fit['intercept'], 13)
        (0.48, 3.62)

        >>> linear_regression([1], [1, 2])
        Traceback (most recent call last):
        ...
        ValueError: Linear regression is defined for vectors of same length

        >>> linear_regression([1], [1])
        Traceback (most recent call last):
        ...
        ValueError: Linear regression is defined for vector of length > 2
    """
    if len(x) != len(y):
        msg = "Linear regression is defined for vectors of same length"
        raise ValueError(msg)
    if len(x) < 2 or len(y) < 2:
        msg = "Linear regression is defined for vector of length > 2"
        raise ValueError(msg)
    xy = [xi * y[i] for i, xi in enumerate(x)]
    mean_xy = mean(xy)
    mean_x = mean(x)
    mean_y = mean(y)
    mean_x2 = mean([math.pow(xi, 2) for xi in x])
    cov = (mean_xy - mean_x * mean_y)
    fit = dict()
    fit['slope'] = -1 * cov / (math.pow(mean_x, 2) - mean_x2)
    fit['intercept'] = mean_y - (fit['slope'] * mean_x)
    return fit
