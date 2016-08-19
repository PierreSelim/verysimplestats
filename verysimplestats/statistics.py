"""Statistics for humans

I lacked a very simple lib for statistics so here it is. It provides really
simplistic float computation statistics indicators.
"""


import math


def mean(x):
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


def median(x):
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

        >> median([])
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


def variance(x):
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


def standard_deviation(x):
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


def covariance(x, y):
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


def correlation(x, y):
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


def linear_regression(x, y):
    """Linear regression based on least sum of square.

    Args:
        x (list): int or float list, explainatory variable
        y (list): int or float list, dependent variable

    Returns:
        LinearRegression: linear regression object

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
    return LinearRegression(x, y)


def ordinary_least_square(x, y):
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


def linear_forecast(slope, intercept, value):
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


def residuals(slope, intercept, x, y):
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


class LinearRegression(object):

    """Simple linear regression (least square estimator)

    Attributes:
        x (list): int or float list, explainatory variable
        y (list): int or float list, dependent variable
        slope (float): slope of the linear regression
        intercept (float): y-intercept of the linear regression (y when x is 0)
        rsquared (float): Coefficient of determination (or r^2)
        residuals (list): residuals of the linear regression (error of the
            model on values used to compute the model)

    Methods:
        linear_forecast(value): apply the linear model on value
    """
    def __init__(self, x, y):
        """Constructor.

        Args:
            x (list): int or float list, explainatory variable
            y (list): int or float list, dependent variable

        Examples:
            >>> LinearRegression([1], [1, 2])
            Traceback (most recent call last):
            ...
            ValueError: Linear regression is defined for vectors of same length

            >>> LinearRegression([1], [1])
            Traceback (most recent call last):
            ...
            ValueError: Linear regression is defined for vector of length > 2

            >>> lm = LinearRegression([1, 2, 3], [1, 3, 4.5])
            >>> lm.x
            [1, 2, 3]
        """
        self.x = x
        self.y = y
        if len(x) != len(y):
            msg = "Linear regression is defined for vectors of same length"
            raise ValueError(msg)
        if len(x) < 2 or len(y) < 2:
            msg = "Linear regression is defined for vector of length > 2"
            raise ValueError(msg)
        self.__slope__ = None
        self.__intercept__ = None
        self.__rsquared__ = None
        self.__residuals__ = None

    @property
    def slope(self):
        """Slope of the linear regression.

        Example:
            >>> lm = LinearRegression([1, 2, 3], [1, 3, 4.5])
            >>> round(lm.slope, 4)
            1.75
        """
        if self.__slope__ is None:
            self.__ordinary_least_square__()
        return self.__slope__

    @property
    def intercept(self):
        """y-Intercept of the linear regression.

        Example:
            >>> lm = LinearRegression([1, 2, 3], [1, 3, 4.5])
            >>> round(lm.intercept, 4)
            -0.6667
        """
        if self.__intercept__ is None:
            self.__ordinary_least_square__()
        return self.__intercept__

    @property
    def rsquared(self):
        """Coefficient of determination

        https://en.wikipedia.org/wiki/Coefficient_of_determination

        Example:
            >>> lm = LinearRegression([1, 2, 3], [1, 3, 4.5])
            >>> round(lm.rsquared, 4)
            0.9932
        """
        if self.__rsquared__ is None:
            self.__rsquared__ = rsquared(self.x, self.y)
        return self.__rsquared__

    def linear_forecast(self, value):
        """Apply the linear model to a given value.

        Args:
            value (float): int or float value

        Returns:
            float: float * slope + intercept

        Example:
            >>> lm = LinearRegression([1, 2, 3], [1, 3, 4.5])
            >>> round(lm.linear_forecast(3), 6)
            4.583333
        """
        return linear_forecast(self.slope, self.intercept, value)

    @property
    def residuals(self):
        """Residuals of the model.

        Returns:
            list: List of residuals (yi - (slope * xi + intercept))

        Example:
            >>> lm = LinearRegression([1, 2, 3], [1, 3, 4.5])
            >>> [round(e, 6) for e in lm.residuals]
            [-0.083333, 0.166667, -0.083333]
        """
        if self.__residuals__ is None:
            self.__residuals__ = residuals(self.slope,
                                           self.intercept,
                                           self.x,
                                           self.y)
        return self.__residuals__

    def __repr__(self):
        """Official representation of the object.

        Example:
            >>> lm = LinearRegression([1, 2], [1, 3])
            >>> repr(lm)
            'LinearRegression([1, 2], [1, 3])'
        """
        return "LinearRegression({x}, {y})".format(x=str(self.x),
                                                   y=str(self.y))

    def __ordinary_least_square__(self):
        """Ordinary least squares

        Tests:
            >>> lm = LinearRegression([1, 2, 3], [1, 3, 4.5])
            >>> lm.__ordinary_least_square__()
            >>> (round(lm.__slope__, 4), round(lm.__intercept__, 4))
            (1.75, -0.6667)
        """
        a, b = ordinary_least_square(self.x, self.y)
        self.__slope__ = a
        self.__intercept__ = b
