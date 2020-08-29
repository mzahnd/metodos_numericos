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


def dec2binf(decNum):
    """Convert decimal to numpy array using IEEE 754 16-bit format.

    Arguments:
        decNum: An intiger or floating point number or one of the following
            strings: 'inf', '+inf' or '-inf', 'nan'.

    Returns:
        Numpy array with size 16.
    """

    expSize = 5
    mantSize = 10
    bias = 2**(expSize-1)-1

    # Sign bit. Can be 0 or 1
    bitSign = 0
    # Exponent and Mantissa arrays
    bitExpArray = np.zeros((expSize,), dtype=int)
    bitMantArray = np.zeros((mantSize,), dtype=int)
    # Set to 1 when converting a subnormal number. 0 otherwise
    subnormal = 0

    # Infinite and NaN array definitions
    inf = np.append([0,
                     1, 1, 1, 1, 1],
                    np.zeros((10,), dtype=int))
    nan = np.append([0,
                     1, 1, 1, 1, 1,
                     1],
                    np.zeros((9,), dtype=int))

    # Perform quick verifications for non numbers
    # Inf and NaN (str)
    if (type(decNum) == str):
        if (decNum.lower() == '-inf'):
            inf[0] = 1
        elif (decNum.lower() != 'inf' and decNum.lower() != '+inf'):
            return nan

        return inf

    # NaN
    if (type(decNum) != int and type(decNum) != float):
        return nan

    # Zero
    if (decNum == 0):
        return np.zeros((16,))

    # Negative number. We'll keep the absolute value.
    if (decNum < 0):
        bitSign = 1
        decNum *= -1

    # Number absolute value is too big
    # Limit calculated using:
    # (-1)^(bitSign)*(2^(30-bias))*(1+2^(-1)+...+2^(-10)+(2^-11)) - 1
    # Taking into account that it's not possible to add 2^(-11) in the
    # mantissa (it wouldn't be a 16 bit number).
    if (decNum > 65519):
        inf[0] = bitSign
        return inf

    # Normal number?
    if (decNum < 2**(1-bias)):
        subnormal = 1

    # Decimal representation of the calculated exponent.
    decExp = 0
    notRoundedDec = 0.0
    # Exponent/Mantissa power that's already been calculated.
    powCalc = 0
    # Exponent bit to calculate next.
    power = 4

    # Calculate exponent
    if (subnormal):
        decExp = 2**(1-bias)
    else:
        # Normal number
        while(power >= 0 and not subnormal):
            # Example to understand this conditional:
            # 2^2 + 2^3 + 2^4 = 2^2 + 24 = 2^power + powerCalc
            if (2**((2**power + powCalc) - bias) <= decNum):
                # Set exponent bit
                bitExpArray[expSize - (power + 1)] = 1
                powCalc += 2**power
                # Calculated exponent in decimal
                decExp = 2**(powCalc - bias)

            power -= 1

    # Skip the loop to calculate the mantissa when the original number has
    # already been calculated. This saves a lot of time.
    if (decNum == decExp):
        power = -11

    # Calculate mantissa
    # power = -1 at this point (unless decNum == decExp)
    powCalc = 0
    while(power >= -10):
        # Remember that subnormal = 0 when the number we're calculating is
        # normal, which is the inverse of what we should add according to the
        # IEEE standard.
        tmpDec = decExp * ((not subnormal) + powCalc + 2**power)

        if (tmpDec <= decNum):
            # Set mantissa bit
            bitMantArray[(power * -1) - 1] = 1
            powCalc += 2**power

            # Whenever a bit has been set, we always have to store the
            # calculated decimal representation of the original number as we
            # don't know where the last bit in the mantissa is and we want to
            # round the number (which is done using the last bit of the
            # mantissa).
            if(power > -10):
                notRoundedDec = tmpDec

        # Round the number setting the last bit when convenient.
        if(power == -10 and
           abs(decNum - tmpDec) < abs(decNum - notRoundedDec)):
            # Set last bit
            bitMantArray[(power * -1) - 1] = 1

        power -= 1

    return np.append([bitSign], np.append(bitExpArray, bitMantArray))


def test():
    """Test dec2binf() function (takes into account numerical rounding).

    16-bit IEEE 754 numbers bits are:
        0    : Sign
        1-5  : Exponent
        6-15 : Mantissa
    """

    # When possible, arrays are created using numpy functions; otherwhise,
    # they are writed as:
    # [sign bit,
    #  exponent bits,
    #  mantissa bits]
    # To make them more human readable.
    testValues = {
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
        65519: np.array([0,
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
        -65519: np.array([1,
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
        "iNF": np.append([0,
                          1, 1, 1, 1, 1],
                         np.zeros((10,), dtype=int)),
        "+inf": np.append([0,
                          1, 1, 1, 1, 1],
                          np.zeros((10,), dtype=int)),
        "-inf": np.append([1,
                          1, 1, 1, 1, 1],
                          np.zeros((10,), dtype=int)),
        # NaN
        "nan": np.append([0,
                          1, 1, 1, 1, 1,
                          1], np.zeros((9,), dtype=int)),
        "NaN": np.append([0,
                          1, 1, 1, 1, 1,
                          1], np.zeros((9,), dtype=int)),
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

    for testNumber in testValues:
        print(f'Testing testNumber: {testNumber}')
        comparison = (dec2binf(testNumber) == testValues[testNumber])
        assert comparison.all(), "The last tested value did not pass the test."


if __name__ == "__main__":
    print("Executed as stand alone script. Running test function.\n")
    test()
