from math import sqrt


def check_only_allowed_chars(test_str, allowed):
    return set(test_str) <= set(allowed)


def is_square(n):
    return sqrt(n).is_integer()


def onehot_city_string(n, l):
    """
    Creates a "onehot" string for one city.

    Parameters
    ----------
    n : integer to encode
    l : length of the string

    Returns
    -------
    s : onehot encoding of n
    """
    assert l>=(n-1)
    return '0'*n + '1' + '0'*(l-n-1)


def format_qaoa_output(s):
    """
    Formats the measured output of QAOA circuit in the "onehot" encoding to
    a list of cities.

    Parameters
    ----------
    G : string with the measurement result

    Returns
    -------
    city_list : list with the cities
    """
    N = len(s)
    assert is_square(N)
    assert check_only_allowed_chars(s,'01')
    city_list = []
    for i in sqrt(N):
        for j in sqrt(N):
            if s[i*N+j] == 1:
                city_list.append(j)
                break
    return(city_list)

def unformat_qaoa_output(city_list):
    """
    Inverse of format_qaoa_output.
    Converts a list of cities to the onehot encoding.

    Parameters
    ----------
    city_list : list with the integer cities

    Returns
    -------
    s : string with the measurement result
    """
    s = ''
    l = len(city_list)
    for c in city_list:
        s += onehot_city_string(c, l)
    return s