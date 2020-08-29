#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trabajo Práctico: Punto Flotante

Created on Mon Aug 24 11:55:55 2020

@author: Batinic Rey, Joaquín
@author: Pompozzi, Magalí M.
@author: Zahnd, Martín E.
"""

import numpy as np


def dec2binf(decimal):
    """Convert decimal to numpy array using IEEE 754 16-bit format.

    Accepts intiger and floating point numbers and 'inf', '+inf' and '-inf'
    as input.

    Return numpy array with size 16.
    """

    exponent_size = 5
    mantissa_size = 10
    bias = 2**(exponent_size-1)-1

    # Sign bit. Can be 0 or 1
    bit_sign = 0
    # Exponent and Mantissa arrays
    bit_exp = np.zeros((exponent_size,), dtype=int)
    bit_mantissa = np.zeros((mantissa_size,), dtype=int)
    # Set to 1 when converting a subnormal number. 0 otherwise
    subnormal = 0

    # Infinite and NaN array definitions
    inf = np.append([0,
                     1, 1, 1, 1, 1],
                    np.zeros((1, 10), dtype=int))
    nan = np.append([0,
                     1, 1, 1, 1, 1,
                     1],
                    np.zeros((1, 9), dtype=int))

    # Perform quick verifications for non numbers
    # Inf and NaN (str)
    if (type(decimal) == str):
        if (decimal == "-inf"):
            inf[0] = 1
        elif (decimal != "inf" and decimal != "+inf"):
            return nan

        return inf

    # NaN
    if (type(decimal) != int and type(decimal) != float):
        return nan

    # Zero
    if (decimal == 0):
        return np.zeros((1, 16))

    # Negative number. We'll keep the absolute value.
    if (decimal < 0):
        bit_sign = 1
        decimal *= -1

    # Number absolute value is too big
    # Limit calculated using:
    # (-1)^(bit_sign)*(2^(30-bias))*(1+2^(-1)+...+2^(-10)+(2^-11)) - 1
    # Taking into account that it's not possible to add 2^(-11) in the mantissa
    if (decimal > 65519):
        inf[0] = bit_sign
        return inf

    # Normal number?
    if (decimal < 2**(1-bias)):
        subnormal = 1

    # Decimal representation of the calculated exponent.
    decimal_exponent = 0
    not_rounded_decimal = 0.0
    # Exponent/Mantissa power that's already been calculated.
    pow_calculated = 0
    # Bit exponent to calculate next.
    power = 4

    # Calculate exponent
    if (subnormal):
        decimal_exponent = 2**(1-bias)
    while(power >= 0 and not subnormal):
        if (2**((2**power + pow_calculated) - bias) <= decimal):
            # Set exponent bit
            bit_exp[5 - (power + 1)] = 1
            pow_calculated += 2**power
            decimal_exponent = 2**(pow_calculated - bias)
        power -= 1

    # Calculate mantissa
    # power = -1 at this point
    pow_calculated = 0
    while(power >= -10 and decimal != decimal_exponent):

        tmp_decimal_number = decimal_exponent \
            * ((not subnormal)
               + pow_calculated
               + 2**power)

        if (tmp_decimal_number <= decimal):
            # Set mantissa bit
            bit_mantissa[(power * -1) - 1] = 1
            pow_calculated += 2**power

            if(power > -10):
                not_rounded_decimal = tmp_decimal_number

        if(power == -10 and
           abs(decimal - tmp_decimal_number)
           < abs(decimal - not_rounded_decimal)):
            bit_mantissa[(power * -1) - 1] = 1

        power -= 1

    final_arr = np.append([bit_sign], np.append(bit_exp, bit_mantissa))
    return final_arr


def test():
    """Test dec2binf() function. Assumes rounding numbers.

    16 bit IEEE 754 numbers bits are:
        0    : Sign
        1-5  : Exponent
        6-15 : Mantissa
    """

    test_values = {
        # Zero
        0: np.zeros((16,), dtype=int),
        # + int
        1022: np.array([0,
                        1, 1, 0, 0, 0,
                        1, 1, 1, 1, 1, 1, 1, 1, 0, 0]),
        1460: np.array([0,
                        1, 1, 0, 0, 1,
                        0, 1, 1, 0, 1, 1, 0, 1, 0, 0]),
        21250: np.array([0,
                         1, 1, 1, 0, 1,
                         0, 1, 0, 0, 1, 1, 0, 0, 0, 0]),
        48740: np.array([0,
                         1, 1, 1, 1, 0,
                         0, 1, 1, 1, 1, 1, 0, 0, 1, 1]),
        65504: np.array([0,
                         1, 1, 1, 1, 0,
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        # - int
        -1022: np.array([1,
                         1, 1, 0, 0, 0,
                         1, 1, 1, 1, 1, 1, 1, 1, 0, 0]),
        -1460: np.array([1,
                         1, 1, 0, 0, 1,
                         0, 1, 1, 0, 1, 1, 0, 1, 0, 0]),
        -21250: np.array([1,
                          1, 1, 1, 0, 1,
                          0, 1, 0, 0, 1, 1, 0, 0, 0, 0]),
        -48740: np.array([1,
                          1, 1, 1, 1, 0,
                          0, 1, 1, 1, 1, 1, 0, 0, 1, 1]),
        -65504: np.array([1,
                          1, 1, 1, 1, 0,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        # + float
        0.0078125: np.array([0,
                             0, 1, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        0.015625: np.array([0,
                            0, 1, 0, 0, 1,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        1.129225162: np.array([0,
                               0, 1, 1, 1, 1,
                               0, 0, 1, 0, 0, 0, 0, 1, 0, 0]),
        65.551895721: np.array([0,
                                1, 0, 1, 0, 1,
                                0, 0, 0, 0, 0, 1, 1, 0, 0, 1]),
        7.401816386: np.array([0,
                               1, 0, 0, 0, 1,
                               1, 1, 0, 1, 1, 0, 0, 1, 1, 1]),
        # - float
        -0.0078125: np.array([1,
                              0, 1, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        -0.015625: np.array([1,
                             0, 1, 0, 0, 1,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        -1.129225162: np.array([1,
                                0, 1, 1, 1, 1,
                                0, 0, 1, 0, 0, 0, 0, 1, 0, 0]),
        -65.551895721: np.array([1,
                                 1, 0, 1, 0, 1,
                                 0, 0, 0, 0, 0, 1, 1, 0, 0, 1]),
        -7.401816386: np.array([1,
                                1, 0, 0, 0, 1,
                                1, 1, 0, 1, 1, 0, 0, 1, 1, 1]),
        # + Subnormal numbers
        0.0000094940765: np.array([0,
                                   0, 0, 0, 0, 0,
                                   0, 0, 1, 0, 0, 1, 1, 1, 1, 1]),
        0.00000807566228: np.array([0,
                                    0, 0, 0, 0, 0,
                                    0, 0, 1, 0, 0, 0, 0, 1, 1, 1]),
        0.000000200928622: np.array([0,
                                     0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 1, 1]),
        0.0000520648489: np.array([0,
                                   0, 0, 0, 0, 0,
                                   1, 1, 0, 1, 1, 0, 1, 0, 0, 1]),
        0.0000000414935258: np.array([0,
                                      0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
        # - Subnormal numbers
        -0.0000094940765: np.array([1,
                                    0, 0, 0, 0, 0,
                                    0, 0, 1, 0, 0, 1, 1, 1, 1, 1]),
        -0.00000807566228: np.array([1,
                                     0, 0, 0, 0, 0,
                                     0, 0, 1, 0, 0, 0, 0, 1, 1, 1]),
        -0.000000200928622: np.array([1,
                                      0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0, 1, 1]),
        -0.0000520648489: np.array([1,
                                    0, 0, 0, 0, 0,
                                    1, 1, 0, 1, 1, 0, 1, 0, 0, 1]),
        -0.0000000414935258: np.array([1,
                                       0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
        # Inf
        65536: np.append([0,
                          1, 1, 1, 1, 1],
                         np.zeros((10,), dtype=int)),
        -65536: np.append([1,
                           1, 1, 1, 1, 1],
                          np.zeros((10,), dtype=int)),
        "inf": np.append([0,
                          1, 1, 1, 1, 1],
                         np.zeros((10,), dtype=int)),
        "+inf": np.append([0,
                          1, 1, 1, 1, 1],
                          np.zeros((10,), dtype=int)),
        "-inf": np.append([1,
                          1, 1, 1, 1, 1],
                          np.zeros((10,), dtype=int)),
        # NaN
        "abcde": np.append([0,
                            1, 1, 1, 1, 1,
                            1], np.zeros((9,), dtype=int)),
        "-abcde": np.append([0,
                             1, 1, 1, 1, 1,
                             1], np.zeros((9,), dtype=int)),
        True: np.append([0,
                         1, 1, 1, 1, 1,
                         1], np.zeros((9,), dtype=int))
    }
    print("Performing tests...")

    for number in test_values:
        print(f'Testing number: {number}')
        comparison = (dec2binf(number) == test_values[number])
        assert comparison.all() == True, "Fail"


if __name__ == "__main__":
    test()
