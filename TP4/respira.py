#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Sources and documentation:

    https://cp-algorithms.com/algebra/binary-exp.html
    https://eli.thegreenplace.net/2009/03/21/efficient-integer-exponentiation-algorithms
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp   # Used for testing
import time
import warnings




def ruku4(f, x0, t0, tf, h):
    step = h
    global t_arr
    t_arr = []
    time_step = t0
    while time_step <= tf:
        t_arr.append(time_step)
        time_step += step

    global x
    x = np.zeros((len(t_arr), np.shape(x0)[0]))

    x[0, :] = x0

    stepOver2 = step / 2
    for k in range(1, len(t_arr)):
        f1 = f(t_arr[k - 1], x[k - 1, :])
        f2 = f(t_arr[k - 1] + stepOver2, x[k - 1, :] + stepOver2 * f1)
        f3 = f(t_arr[k - 1] + stepOver2, x[k - 1, :] + stepOver2 * f2)
        f4 = f(t_arr[k - 1] + step, x[k - 1, :] + step * f3)

        xnew = step * (f1 + 2 * f2 + 2 * f3 + f4) / 6
        x[k, :] = x[k - 1, :] + xnew

    #print("ruk4: Done.")
    return t_arr, x


def closest(lst, K):
    """Python3 program to find Closest number in a list."""
    return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - K))]


def mackeyglass_dxdt(t, x_org):
    tMinus02 = t - 0.2
    xt02 = 0
    
    if tMinus02 > 0:
        closest_time = closest(t_arr, tMinus02)
        for index, element in enumerate(t_arr):
            if np.isclose(element, closest_time):
                xt02 = x[index]
                break
    else:
        xt02 = 0.5
    
    xt02FiveTimes = powerint(float(xt02), 5)
    dxdt = 6 - 16 * x_org * xt02FiveTimes / (1 + xt02FiveTimes)

    return dxdt


def errorCalc(f, x0, t0, tf, h, obtainedX):
    # HS means Half step -> x Half step: Xs obtained with a step half of the size

    _, xDoubleStep = ruku4(f, x0, t0, tf, h / 2)

    errk = np.zeros(0)

    for i in range(obtainedX.shape[1]):
        error = (xDoubleStep[2 * i] - obtainedX[i]) / 31
        np.append(errk, error)

    return errk


def mackeyglass():
    x0 = np.array([0.5])
    t0 = 0
    tf = 5
    step = tf / 1000
   
    # print(f"xrk4: {xrk4}") # xrk4: [[0.]]
    # errorrk4 = errorCalc(mackeyglass_dxdt, x0, t0, tf, step, xrk4)

    t, xrk4 = ruku4(mackeyglass_dxdt, x0, t0, tf, step)

    # Graph
    print("Plotting...")
    _, axes = plt.subplots()
    axes.plot(t, xrk4[:], label='RuKu 4')
    plt.title('Mackey-Glass')
    axes.legend()

    #print("Calculation Error:")
    # print(errorrk4)

    try:
        print("Showing plot.")
        plt.show()
    except KeyboardInterrupt:
        print("Closing plot...")
        plt.close('all')

    print("mackeyglass: Done.")


def powerint (x, p):
    """Calculate base^exponent (x^p)

    When p < 0, x^(|p|) is evaluated first and then inverted.
    This way no precision is lost due to floating point during the
    exponentiation part of the process.

    Arguments:
        x: Intiger or floating point number greater or equal to zero.
        p: Intiger number (different than zero iff x equals zero).

    Returns:
        The solution of x^p

    Raises:
        RuntimeError
        TypeError
        ArithmeticError
        ZeroDivisionError
        OverflowError
    """
    
    # Change the arguments name for easier reading through the code
    base = x
    exponent = p

    # Check that both arguments are valid.
    if base < 0:
        print(f'Your input:\n\tBase: {base}\n\tExponent: {exponent}')
        raise RuntimeError(
            "Please provide a non negative number for the base.")
    elif (type(exponent) is not int or not np.intc or not np.int_ \
            or not np.int8 or not np.int16 or not np.int32 or not np.int64) \
            and (type(exponent) is not type(base)):
        raise TypeError("Exponent must be an int.")
    elif base == 0:
        if exponent == 0:
            raise ArithmeticError("Math error. Trying to perform: 0^0.")
        elif exponent < 0:
            raise ZeroDivisionError("Math error. Trying to divide by zero.")

    # Powers that are not necessary to calculate: 0^p, 1^p, x^0, x^1
    if base == 0:
        return 0
    elif base == 1 or exponent == 0:
        return 1
    elif exponent == 1:
        return base

    # Warning management
    np.seterr(all='warn')
    with warnings.catch_warnings():
        warnings.simplefilter('error')

    # Negative exponent management
    negativeExponent = False
    if exponent < 0:
        negativeExponent = True
        exponent *= -1
  
    result = 0

    try:
        result = _binaryExponent(base, exponent)

    except Warning:
        if negativeExponent is True:
            exponent *= -1
        raise OverflowError("Overflow while performing: ", 
                            f"{base})^({exponent}).")

    if negativeExponent is True and result != 0:
        return float(1/result)
    else:
        return result


def _binaryExponent(base, exponent):
    # For an exponent <= 4 the algorithm makes no sense.
    # log_2(4) = 2 and 4 = 0b100, which will result in the exact same 
    # process as the binary exponentiation algorithm.
    #
    # Avoiding it, we are actually saving a few operations (the bitwise and the
    # comparison).

    result = 1

    if (exponent == 0 and base == 0):
        raise RuntimeError("Magic error happened: 0^0.")
    elif (exponent == 0 and base != 0):
        return 1
    elif (exponent < 5):
        for _ in range(exponent):
            result *= base
    else:
        while (exponent > 0):
            # Current bit = 1
            if (exponent & 1):
                result *= base
            base *= base
            exponent = exponent >> 1
    
    return result


def test():
    test_ruku4()
    test_powerint()


def test_powerint():
    # Valid combinations of x^p powerint

    # For powerint
    simpleIntPowers = (
        ([0, 1]), ([0, 5]), ([0, 17]), ([0, 752]),
        ([2, 0]), ([1, 0]), ([1435, 0]), ([523, 0]),
        ([1, 1]), ([1, 5]), ([1, -5223]), ([1,-3]),
        ([2, 2]), ([2, 15]), ([2, 5]),
        ([2, -5]), ([2, -58]),
        # Some random-generated sets (using random.org)
        ([51, -10]),            ([5403, -3]),
        ([43, 36]),             ([95, 8]),
        ([27, 10]),             ([32, 5]),
        ([76, -12]),            ([67, 10]),
        ([31, 9]),              ([57, -9]),
        ([17, 14]),             ([38, -60]),
        ([52, 7]),              ([10, -8]),
        ([28, -25]),            ([69, 4]),
        ([67, 10]),             ([75, 3]),
        ([93, 8]),              ([41, -2]),
        ([24, -4]),             ([37, 9]),
        ([82, -38]),            ([48, -5]),
        ([18, -17]),            ([50, -19]),
        ([2, 20]),              ([7, -3]),
        ([90, 20]),             ([592, 94]),
        ([819645, -369723]),    ([22962, 396793]),
        ([423837, -78785]),     ([562804, 447158]),
        ([506033, 1233]),       ([864805, -7011]),
        ([382783, -713864]),    ([873793, -974258]),
        ([381540, -262639]),    ([152469, -173293]),
    )


    simpleFloatPowers = (
      ([0.0, 1]), ([0.0, 17]),
      ([0.0, 752]),
      ([2.0, 0]), ([1.0, 0]),
      ([1435.0, 0]),
      ([523.0, 0]),
      ([1.0, 1]), ([1.0, 5]),
      ([1.0, -5223]),
      ([1.0, -3]),
      ([2.0, 2]), ([2.0, 15]),
      ([2.0, -5]), ([2.0, -58]),
      # Some random-generated sets (using random.org)
      ([22652.6, 30]), ([30.55, -37]),
      ([902.599, -2]), ([385.349, 47]),
      ([70.846, 91]), ([28.341, 21]),
      ([51.886, -16]), ([543.81, -69]),
      ([43.19, -36]), ([95.331, 8]),
      ([271.991, 10]), ([3220.9, -25]),
      ([76.57, -12]), ([63.973, 12]),
      ([31.928, 9]), ([59.695, -9]),
      ([137.350, 14]), ([338.48, -60]),
      ([542.261, 17]), ([10.866, -8]),
      ([28.265, -25]), ([569.852, 4]),
      ([67.249, 36]), ([75.968, 3]),
      ([923.4, 81]), ([414.84, 2]),
      ([24.15, -4]), ([370.314, 9]),
      ([82.880, -38]), ([48.97, -5]),
      ([188.91, -17]), ([50.177, -19]),
      ([2.360, 20]), ([7.376, -3]),
      ([90.424, 20]), ([59.912, -9]),
    )

    numberString = ''

    # Tests for powerint
    print('='*60, "\nTesting powerint()...\n", '-'*59)
    # Invalid combinations of x^p
  
    # Valid integer combinations of x^p
    for base, exponent in simpleIntPowers:
        numberString = f'({base})^({exponent})'
        print ("Testing: {0:<30}\t\t".format(numberString), end=' ')
        test_powerint_assert(base, exponent)
        print("-> Pass <-")
  
    # Valid floating point combinations of x^p (p is always integer)
    for base, exponent in simpleFloatPowers:
        numberString = f'({base})^({int(exponent)})'
        print ("Testing: {0:<30}\t\t".format(numberString), end=' ')
        test_powerint_assert(base, exponent, dataType=float)
        print("-> Pass <-")


def test_powerint_assert(base, exponent, dataType=int):
    """Test case for powerint function.

    Calls powerint and compares its answer with python's exponentiation
    function.
    Whenever the exponent is a negative number, numpy's allclose function
    is used instead to deal with floating point numbers.

    This function will stop the code execution when it finds an error.

    Returns:
        None

    Raises:
        Nothing
    """
  
    ans = powerint(base, exponent)
    #print(ans)

    # This is not meant to be fast, but to be simple and not failing.
    if base == 0:
        pyPow = 0
    elif exponent < 0 and base != 0:
        pyPow = 1/(base**((-1)*exponent))   
    else:
        pyPow = base**exponent
  
    # For debuging purposes only
    #print('{0} || {1}'.format(ans, pyPow))

    #assert True, "X ERROR X"
    if dataType == int and exponent > 0:
        assert ans == pyPow, "X ERROR X"
    else:
        assert np.allclose(ans, pyPow), "X ERROR X"


def test_ruku4():
    """We'll compare our answers with the ones from scipy.integrate.solve_ivp 
    using RK45 method."""

    # Some example functions were taken from:
    # http://epsem.upc.edu/~fpq/minas/pr-sol/prob-edos-n-solu
    #
    # Or were given during the corresponding lecture.
    
    # Step is calculated as the second element of 'interval'/'step_divisor'
    functionsAndParams = (
        {
            'function': ['cos(t)', lambda t, x: np.cos(t)],
            'interval': [-100, 10],
            'step_divisor': 10000,
            'x0': np.zeros(1)
        },
        {
            'function': ['sin(t)', lambda t, x: np.sin(t)],
            'interval': [-10, 80],
            'step_divisor': 10000,
            'x0': np.array([np.pi])
        },
        {
            'function': ['ln(t)*x', lambda t, x: np.log(t)*x],
            'interval': [0.1, 50],
            'step_divisor': 10000,
            'x0': np.array([-1])
        },
        {
            'function': ['x*e^(t)', lambda t, x: x*np.exp(t)],
            'interval': [-5, 5],
            'step_divisor': 10000,
            'x0': np.array([0.458])
        },
        {
            'function': ['5*e^(-t)+0.5*e^(-2t)-e^(-2t)*sin(e^t)', 
                lambda t, x: 
                5*np.exp(-t)+np.exp(-2*t)*(0.5-np.sin(np.exp(t)))],
            'interval': [-5, 5],
            'step_divisor': 10000,
            'x0': np.zeros(1)
        },
        {
            'function': ['1.64*e^(-t)-5.45te^(-t)+e^(-t)*(t^2)*(ln(t)/2-3/4)', 
                lambda t, x: np.exp(-t)*(1.64-5.45*t+t*t*(np.log(t)/2-3/4))],
            'interval': [0.01, 10],
            'step_divisor': 10000,
            'x0': np.zeros(1)
        },
        {
            # This function is taken from an RC circuit.
            # dx = (A*cos(w*t)-x) / (R*C)
            # Where, A = 1.0; w = 2*pi*1000; R = 1e3 and C = 1e-6
            'function': ['(1.0*cos(2*pi*1000*t)-x)/(1e3*1e-6)',
                lambda t, x: ((np.cos(np.pi*2000*t)-x)*1000)],
            'interval': [0, 0.005],
            'step_divisor': 1000,
            'x0': np.zeros(1)
        },
    )

    print('='*60, "\nTesting ruku4()...\n", '-'*59)

    for fun in functionsAndParams:
        fname = fun['function'][0]
        x0 = fun['x0']
        t0 = fun['interval'][0]
        tf = fun['interval'][1]
        step = fun['interval'][1]/fun['step_divisor']

        print('Comparing:')
        print(f'\tf(t, x) = {fname}')
        print(f'\tIn [{t0} ; {tf}], using a step = {step} and x0 = {x0}')
        
        #print('Computing function using ruku4...', end=' ')
        tRuku4, xRuku4 = ruku4(fun['function'][1], x0, t0, tf, step)
        #print('Done.')

        #print('Computing function using solve_ivp at the same points...', 
        #        end=' ')
        rk45 = solve_ivp(fun['function'][1],[t0,tf], x0, method='RK45',
                        t_eval=tRuku4, rtol=1e-13, atol=1e-14)
        #print('Done.')


        #print('Starting comparison between both answers...')
        equalX = compareOutputs(xRuku4, rk45.y.T)
        #print('Done.', end=' ')
        if not equalX:
            print('X FAIL X')
            print('Errors were found while comparing both functions.')
        else:
            print('-> Pass <-')
            
    #plotRuku4vsRK45(xRuku4, tRuku4, rk45.y, rk45.t)


def compareOutputs(ruku4_x, rk45_x):
    """Compare the output between our ruku4 and an already implemented RK45."""

    allSimilar = True
    for i in range(ruku4_x.size):
        try:
            ans = math.isclose(ruku4_x[i], rk45_x[i],
                                rel_tol=1e-03, abs_tol=1e-05)
            if not ans:
                print(f'False: {ruku4_x[i]} || {rk45_x[i]}')
                allSimilar = False

        except IndexError:
            print('The computed value was too large.', 
            'Try reducing the interval for this function.')
            return False

    return allSimilar
    

def plotRuku4vsRK45(ruku4_x, ruku4_t, rk45_x, rk45_t, title='Test plot.'):
    # Graph
    print("Plotting...")
    _, axes = plt.subplots()
    axes.plot(ruku4_t, ruku4_x[:], label='RuKu 4')
    axes.plot(rk45_t, rk45_x[0,:], label='RK5')
    plt.title(title)
    axes.legend()
    

    print("Showing plot.")
    plt.show()


if __name__ == "__main__":
    print("Running...")
    initialTime = time.time_ns()
    test()
    #mackeyglass()

    finalTime = time.time_ns()

    print("\nRan in: %f s (%d ns)" % ((finalTime-initialTime)*10**(-9),
                                    finalTime-initialTime))


    #mackeyglass()
    #print("Main code: Done.")