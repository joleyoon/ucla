import math
from collections import Counter

def gcd(x, y):
    """
    Calculates the greatest common divisor (GCD) of two integers.
    """
    x, y = abs(int(x)), abs(int(y))
    while x != 0 and y != 0:
        if x > y:
            x = x % y
        else:
            y = y % x
    return max(x, y)


def gcdi(x):
    """
    Calculates the GCD for a list of integers.
    """
    if len(x) < 2:
        raise ValueError("Input list must have at least two elements.")
    
    result = x[0]
    for i in range(1, len(x)):
        result = gcd(result, x[i])
    return result


def lcm(x, y):
    """
    Calculates the least common multiple (LCM) of two integers.
    """
    return abs(x * y) // gcd(x, y)


def add_2_frac(n1, d1, n2, d2):
    """
    Adds two fractions and returns the result as a dict with numerator and denominator.
    """
    d = lcm(d1, d2)
    n1 = n1 * (d // d1)
    n2 = n2 * (d // d2)
    numerator = n1 + n2
    
    return {"num": numerator, "denom": d}


def is_prime(x):
    """
    Checks whether elements in a list are prime numbers.
    Returns a list of booleans.
    """
    result = []
    for element in x:
        if element <= 1:
            result.append(False)
        elif element == 2:
            result.append(True)
        else:
            prime = True
            for divisor in range(2, int(math.sqrt(element)) + 1):
                if element % divisor == 0:
                    prime = False
                    break
            result.append(prime)
    return result


def get_factors(x):
    """
    Returns the prime factors of a number along with their exponents.
    """
    if x <= 1:
        raise ValueError("Input must be greater than 1")
    if not isinstance(x, int):
        raise ValueError("Input must be an integer")
    
    all_factors = []
    divisor = 2
    while x > 1:
        while x % divisor == 0:
            all_factors.append(divisor)
            x //= divisor
        divisor += 1
        if divisor * divisor > x and x > 1:
            all_factors.append(x)
            break
    
    prime_counts = Counter(all_factors)
    primes = list(prime_counts.keys())
    exponents = list(prime_counts.values())
    
    return {"primes": primes, "exponents": exponents}