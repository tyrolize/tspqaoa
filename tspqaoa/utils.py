from math import sqrt


def check_only_allowed_chars(test_str, allowed):
    return set(test_str) <= set(allowed)

def is_square(n):
    return sqrt(n).is_integer()

def format_qaoa_output(s):
    N = len(s)
    assert is_square(N)
    assert check_only_allowed_chars(s,'01')
    city_sequence = []
    for i in sqrt(N):
        for j in sqrt(N):
            if s[i*N+j] == 1:
                city_sequence.append(j)
                break
    return(city_sequence)