from functools import partial
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
    assert int(l)>=(int(n)-1)
    return '0'*int(n) + '1' + '0'*(l-int(n)-1)


def format_from_onehot(s, translate=None):
    """
    Formats the measured output of QAOA circuit in the "onehot" encoding to
    a list of cities.

    Parameters
    ----------
    s : string with the measurement result
    translate : dictionary with city correspondencies (1..k) to graph notation

    Returns
    -------
    city_list : list with the cities
    """
    N = len(s)
    assert is_square(N)
    assert check_only_allowed_chars(s,'01')
    city_list = []
    for i in range(int(sqrt(N))):
        for j in range(int(sqrt(N))):
            if s[i*int(sqrt(N))+j] == '1':
                city_list.append(j)
                break
    if translate:
        translated_city_list = [x if x not in translate else translate[x] for x in city_list]
        return(translated_city_list)
    else:
        return(city_list)

def unformat_to_onehot(city_list, translate=None):
    """
    Inverse of format_qaoa_output.
    Converts a list of cities to the onehot encoding.

    Parameters
    ----------
    city_list : list with the integer cities
    translate : dictionary with city correspondencies graph notation to (1..k)

    Returns
    -------
    s : string with the measurement result
    """
    s = ''
    l = len(city_list)
    if translate:
        translated_city_list = [x if x not in translate else translate[x] for x in city_list]
        for c in translated_city_list:
            s += onehot_city_string(c, l)
    else:
        for c in city_list:
            s += onehot_city_string(c, l)
    return s