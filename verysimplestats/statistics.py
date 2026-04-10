"""Statistics for humans

I lacked a very simple lib for statistics so here it is. It provides really
simplistic float computation statistics indicators.
"""


import math
from dataclasses import dataclass


def mean(x: list[float]) -> float:
    """Mean.

    Args:
        x (list): float (int) list

    Returns:
        float: Mean value of x.

    Raises:
        ValueError: when x is empty

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
        TypeError: object of type 'int' has no len()
    """
    if len(x) < 1:
        raise ValueError("An empty data list has no mean {x}".format(x=x))
    return math.fsum(x) / len(x)


def median(x: list[float]) -> float:
    """Median.

    When x has an even number of element, mean of the two central value of the
    distribution.

    Args:
        x (list): float (int) list

    Returns:
        float: Median value of x

    Raises:
        ValueError: when x is empty

    Examples:
        >>> median([5, 2, 4, 1, 3])
        3.0

        >>> median([5, 2, 6, 4, 1, 3])
        3.5

        >>> median([])
        Traceback (most recent call last):
        ...
        ValueError: An empty data list has no median []

        >>> median(42)
        Traceback (most recent call last):
        ...
        TypeError: object of type 'int' has no len()
    """
    if len(x) < 1:
        raise ValueError("An empty data list has no median {x}".format(x=x))
    y = sorted(x)
    middle = len(x) / 2
    if len(x) % 2 == 0:
        return (y[int(middle - 0.5)] + y[int(middle + 0.5)]) / 2.0
    else:
        return float(y[int(middle)])


def variance(x: list[float]) -> float:
    """Unbiased sample variance.

    The population variance is (n-1)/n * variance(x)

    Args:
        x (list): float (int) list

    Returns:
        float: Unbiased sample variance

    Raises:
        ValueError: when length of x is strictly lower than 2

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


def standard_deviation(x: list[float]) -> float:
    """Unbiased sample standard deviation.

    Args:
        x (list): float (int) list

    Returns:
        float: Unbiased sample standard deviation

    Raises:
        ValueError: when length of x is strictly lower than 2

    Examples:
        >>> round(standard_deviation([1, 2, 3, 4, 5]), 13)
        1.5811388300842
    """
    return math.sqrt(variance(x))


def covariance(x: list[float], y: list[float]) -> float:
    """Unbiased sample covariance.

    Args:
        x (list): float or int list
        y (list): float or int list

    Returns:
        float: Unbiased sample covariance

    Raises:
        ValueError: when x and y do not have the same length or when x or y are
            empty

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


def correlation(x: list[float], y: list[float]) -> float:
    """Pearson's correlation coefficient.

    Args:
        x (list): float or int list
        y (list): float or int list

    Returns:
        float: Pearson's correlation coefficient

    Raises:
        ValueError: when x and y do not have the same length or when length of
            x or y is strictly lower than 2

    Examples:
        >>> round(correlation([1, 2, 3], [-1, -2, -3]), 13)
        -1.0

        >>> correlation([-2, -1, 0, 1, 2], [2, 1, 0, 1, 2])
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


def rsquared(x: list[float], y: list[float]) -> float:
    """Coefficient of determination (also known as R^2)

    https://en.wikipedia.org/wiki/Coefficient_of_determination

    Args:
        x (list): int or float list, explainatory variable
        y (list): int or float list, dependent variable

    Returns:
        float: Coefficient of determination

    Raises:
        ValueError: when x and y do not have the same length or when length of
            x (or y) is strictly lower than 2

    Examples:
        >>> round(rsquared([1, 2, 3], [-1, -2, -3]), 13)
        1.0
    """
    return math.pow(correlation(x, y), 2)


@dataclass
class LinearRegressionFit:
    """Result of a linear regression fit."""

    slope: float
    intercept: float
    rsquared: float
    residuals: list[float]


def linear_regression(x: list[float], y: list[float]) -> LinearRegressionFit:
    """Linear regression based on least sum of square.

    Args:
        x (list): int or float list, explainatory variable
        y (list): int or float list, dependent variable

    Returns:
        LinearRegressionResult: dataclass with slope, intercept, rsquared, residuals

    Raises:
        ValueError: when x and y do not have the same length or when length of
            x (or y) is strictly lower than 2

    Examples:
        >>> lm = linear_regression([1, 2, 3], [1, 3, 4.5])
        >>> round(lm.slope, 13), round(lm.intercept, 13)
        (1.75, -0.6666666666667)

        >>> linear_regression([1], [1, 2])
        Traceback (most recent call last):
        ...
        ValueError: Linear regression is defined for vectors of same length

        >>> linear_regression([1], [1])
        Traceback (most recent call last):
        ...
        ValueError: Linear regression is defined for vector of length > 2
    """
    slope, intercept = ordinary_least_square(x, y)
    return LinearRegressionFit(
        slope=slope,
        intercept=intercept,
        rsquared=rsquared(x, y),
        residuals=residuals(slope, intercept, x, y),
    )


def ordinary_least_square(x: list[float], y: list[float]) -> tuple[float, float]:
    """Ordinary least squares.

    Args:
        x (list): int or float list, explainatory variable
        y (list): int or float list, dependent variable

    Returns:
        float, float: Couple with slope and y-intercept of the OLS linear fit

    Raises:
        ValueError: when x and y do not have the same length or when length of
            x (or y) is strictly lower than 2

    Tests:
        >>> slope, intercept = ordinary_least_square([1, 2, 3], [1, 3, 4.5])
        >>> (round(slope, 4), round(intercept, 4))
        (1.75, -0.6667)

        >>> ordinary_least_square([1], [1, 2])
        Traceback (most recent call last):
        ...
        ValueError: Linear regression is defined for vectors of same length

        >>> ordinary_least_square([1], [1])
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
    cov = (mean_xy - mean_x * mean_y)  # biased indicator
    slope = -1 * cov / (math.pow(mean_x, 2) - mean_x2)
    intercept = mean_y - (slope * mean_x)
    return slope, intercept


def linear_forecast(slope: float, intercept: float, value: float) -> float:
    """Apply the linear model to a given value.

    Args:
        slope (float): slope of the linear model
        intercept (float): intercept of the linear model
        value (float): int or float value

    Returns:
        float: slope * x + intercept

    Example:
        >>> linear_forecast(2, -1, 3)
        5.0
    """
    return slope * float(value) + intercept


def residuals(slope: float, intercept: float, x: list[float], y: list[float]) -> list[float]:
    """Residuals of the linear regression

    Args:
        slope (float): slope of the linear model
        intercept (float): intercept of the linear model
        x (list): int or float list, explainatory variable
        y (list): int or float list, dependent variable


    Returns:
        list: List of residuals [ei = yi - (slope * xi + intercept)]
    """
    return [y[i] - linear_forecast(slope, intercept, xi)
            for i, xi in enumerate(x)]
