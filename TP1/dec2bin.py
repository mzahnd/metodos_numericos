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
    if decimal == 0:
        return np.zeros((1, 16), dtype=int)


def test():
    """Test dec2binf() function. Assumes rounding numbers.

    16 bit IEEE 754 numbers bits are:
        0    : Sign
        1-5  : Exponent
        6-15 : Mantissa
    """
    
    test_values = {
        # Zero
        0: np.zeros((1, 16), dtype=int),
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
        0.0000000414935258: np.append(np.zeros((1, 14), dtype=int), [1]),
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
        -0.0000000414935258: np.append([1],
                                       np.append(np.zeros((1, 13), dtype=int),
                                       [1])),
        # Inf
        65536: np.append([0,
                          1, 1, 1, 1, 1],
                         np.zeros((1, 10), dtype=int)),
        -65536: np.append([1,
                           1, 1, 1, 1, 1],
                          np.zeros((1, 10), dtype=int))
    }
    print("Performing tests...")

    for number in test_values:
        print(f'Testing number: {number}')
        comparison = dec2binf(number) == test_values[number]
        assert comparison.all() == True, "Fail"


if __name__ == "__main__":
    test()
