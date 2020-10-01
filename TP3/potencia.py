#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trabajo Práctico: Ecuaciones no-lineales

@author: Batinic Rey, Joaquín
@author: Pompozzi, Magalí M.
@author: Zahnd, Martín E.

Sources:
  Warning management:
  https://stackoverflow.com/a/15934081

  Why using int and float as datatypes and not np.uint64 or similar:
  https://www.python.org/dev/peps/pep-0237/
  https://docs.python.org/3/c-api/long.html
  More on this thread: https://stackoverflow.com/q/538551
"""

import math
import numpy as np
import time
import warnings

# Tolerance of the bisection method used in powerrat()
_POWERRAT_TOLERANCE=0.000001


def powerint (x, p):
  """Calculate x^p

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
    OverflowError
  """

  # Check that both arguments are valid.
  if x < 0:
    raise RuntimeError("Please provide a non negative number for the base.")
  elif type(p) is not int or not np.intc or not np.int_ or not np.int8 \
                or not np.int16 or not np.int32 or not np.int64:
    raise TypeError("Exponent must be an int.")
  elif (x == 0 and p == 0):
    raise ArithmeticError("Math error. Trying to perform: 0^0")

  # Powers that are not necessary to calculate: 0^p, 1^p, x^0, x^1
  if x == 0:
    return 0
  elif x == 1 or p == 0:
    return 1
  elif p == 1:
    return x

  negativeExponent = False
  y = int(1)
  
  if p < 0:
    negativeExponent = True
    p *= -1
  
  if p != 0:
    np.seterr(all='warn')
    with warnings.catch_warnings():
      warnings.simplefilter('error')

      for _ in range(int(p)):
        try:
          y *= x

        except Warning:
          if negativeExponent == True:
            p *= -1
          raise OverflowError(f"Overflow while performing: ({x})^({p}).")
      
  if negativeExponent == True and y != 0:
    return float(1/y)
  else:
    return y


def _bisection(point, q, y):
  """This function is used by powerrat to evaluate a possible bisection point.

  The original equation is:
    point = y^(1/q)
  Which equals to, as long as point > 0:
    (point^q) - y = 0

  Returns:
    (point^q) - y

  Raises:
    Noting
  """

  '''Esta función es la que se evalúa para despejar 
  por bisección point = y^(1/q)'''
  return powerint(point, q) - y

def powerrat (x, p, q):
  """Approximate the solution of x^(p/q) with tolerance _POWERRAT_TOLERANCE.

  Arguments:
    x: Intiger or floating point number greater or equal to zero.
    p: Intiger number
    q: Intiger number different than zero.

  Returns:
    An approximation of the solution of x^(p/q), with tolerance 
    _POWERRAT_TOLERANCE

  Raises:
    RuntimeError
    TypeError
    ArithmeticError
    ZeroDivisionError
  """

  # Verify that all arguments are valid
  if x < 0:
    raise RuntimeError("Please provide a non negative number for the base.")
  elif (type(p) is not int or not np.intc or not np.int_ or not np.int8 \
                or not np.int16 or not np.int32 or not np.int64) \
       and (type(p) is not type(q)):
    raise TypeError("Exponent must be an int.")
  elif (x == 0 and p == 0):
    raise ArithmeticError("Math error. Trying to perform: 0^0")
  elif (q == 0):
    raise ZeroDivisionError("Math error. q must not be zero.")

  # Powers that are not necessary to calculate: 0^(p/q), 1^(p/q) x^0, x^1
  if p == q:
    return x
  elif (x != 0 and p == 0) or (x == 1):
    return 1
  elif x == 0:
    return 0

  #    (-)*(-) = (+)     Invert p and q signs
  if (p < 0 and q < 0) or (p > 0 and q < 0):
    p *= -1
    q *= -1


  # The case where p/q is an integer number must be taken into 
  # consideration as it could save a lot of time.
  # This can be tested evaluating, for example, (67)^(10/2) with and
  # without the following lines in this function.
  possibleRealExponent = p/q
  if possibleRealExponent.is_integer():
    p = int(possibleRealExponent)
    q = 1

  y = powerint(x, p)

  # In this case, it's like calling powerint directly.
  # There's nothing more to do
  if q == 1:
    return y

  # First interval
  a = 0
  b = 10
  # Expand the interval if f(a)f(b)>0
  while _bisection(a,q,y)*_bisection(b,q,y) > 0:
    b += 1000

  # Perfom bisection method for finding the root
  for _ in range(10000):
    c = (a + b)/2
    if (b - a) > _POWERRAT_TOLERANCE:
      if _bisection(a,q,y)*_bisection(c,q,y) > 0:
        a = c
      else:
        b = c
    else:
      return c
  return c


def test():
  """Test powerint and powerrat functions.

  As powerrat depends on powerint, the last one is tested on the first
  place.
  
  This function will stop the code execution when it finds an error.

  Returns:
    None

  Raises:
    Nothing
  """

  # Valid combinations of x^p and x^(p/q) for both, powerint and powerrat
  
  # For powerint
  simpleIntPowers = (
    ([0, 1]), ([0, 17]), ([0, 752]),
    ([2, 0]), ([1, 0]), ([1435, 0]), 
    ([523, 0]),
    ([1, 1]), ([1, 5]), ([1, -5223]),
    ([1,-3]),
    ([2, 2]), ([2, 15]), ([2, -5]),
    ([2, -58]),
    # Some random-generated sets (using random.org)
    ([2265, 3]), ([20, -3]), 
    ([902, -2]), ([385, 7]),
    ([70, 9]), ([28, 12]),
    ([51, -10]), ([5403, -3]),
    ([43, 36]), ([95, 8]),
    ([27, 10]), ([32, 5]),
    ([76, -12]), ([67, 10]),
    ([31, 9]), ([57, -9]),
    ([17, 14]), ([38, -60]),
    ([52, 7]), ([10, -8]),
    ([28, -25]), ([69, 4]),
    ([67, 10]), ([75, 3]),
    ([93, 8]), ([41, -2]),
    ([24, -4]), ([37, 9]),
    ([82, -38]), ([48, -5]),
    ([18, -17]), ([50, -19]),
    ([2, 20]), ([7, -3]),
    ([90, 20]), ([59, -9]),
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

  # For powerrat
  rationalIntPowers = (
    ([0, 1, 1]), ([0, 17, 17]), ([0, 1, 752]),
    ([2, 0, 1]), ([1, 0, -5]), ([1435, 0, 9]), 
    ([523, 0, -98]),
    ([1, 5, 1]), ([1, -5, 1]), ([1, -5223, -8]),
    ([1, -3, 4]),
    ([2, 2, 2]), ([2, 15 ,3]), ([2, -5, 96]),
    ([2, -58, 29]),
    # Some random-generated sets (using random.org)
    ([2265, 3, 2]), ([20, -3, 16]), 
    ([902, -2, 89]), ([385, 7, 81]),
    ([70, 9, 40]), ([28, 12, -91]),
    ([51, -10, -59]), ([5403, -3, 12]),
    ([43, -36, 99]), ([95, 8, 92]),
    ([27, 10, -55]), ([32, 5, -5]),
    ([76, -12, 36]), ([67, 10, -25]),
    ([31, 9, 72]), ([57, -9, 46]),
    ([17, 14, 40]), ([38, -60, -87]),
    ([52, 7, 6]), ([10, -8, 64]),
    ([28, -25, 94]), ([69, 4, 23]),
    ([67, 10, 2]), ([75, 3, 8]),
    ([93, 18, -17]), ([41, -2, 90]),
    ([24, -4, 86]), ([37, 9, -55]),
    ([82, -38, 41]), ([48, -5, 20]),
    ([18, -17, 28]), ([50, -19, 1]),
    ([2, 20, 40]), ([7, -3, 27]),
    ([90, 20, -93]), ([59, -9, 40]),
  )

  rationalFloatPowers = (
    ([0.0, 1, 1]), ([0.0, 17, 17]), ([0.0, 1, 752]),
    ([2.0, 0, 1]), ([1.0, 0, -5]), ([1435.0, 0, 9]), 
    ([523.0, 0, -98]),
    ([1.0, 5, 1]), ([1.0, -5, 1]), ([1.0, -5223, -8]),
    ([1.0, -3, 4]),
    ([2.0, 2, 2]), ([2.0, 15, 3]), ([2.0, -5, 96]),
    ([2.0, -58, 29]),
    # Some random-generated sets (using random.org)
    ([22652.6, 3, 2]), ([30.55, -3, 16]), 
    ([902.599, -2, 89]), ([385.349, 7, 81]),
    ([70.846, 9, 40]), ([28.341, 12, -91]),
    ([51.886, -10, -59]), ([543.81, -3, 12]),
    ([43.19, -36, 99]), ([95.331, 8, 92]),
    ([271.991, 10, -55]), ([32.9, 5, -5]),
    ([76.57, -12, 36]), ([63.973, 10, -25]),
    ([31.928, 9, 72]), ([59.695, -9, 46]),
    ([137.350, 14, 40]), ([338.48, -60, 87]),
    ([542.261, 7, 6]), ([10.866, -8, 64]),
    ([28.265, -25, 94]), ([569.852, 4, 23]),
    ([67.249, 10, 2]), ([75.968, 3, 8]),
    ([93.4, 18, -17]), ([414.84, -2, 90]),
    ([24.15, -4, 86]), ([370.314, 9, -55]),
    ([82.880, -38, 41]), ([48.97, -5, 20]),
    ([188.91, -17, 28]), ([50.177, -19, 1]),
    ([2.360, 20, 40]), ([7.376, -3, 27]),
    ([90.424, 20, -93]), ([59.912, -9, 40]),
  )

  # Invalid conditions that should raise an error
  # For powerint
  powerintInvalids = (
    ([0, 0]), ([-5, 0]), ([-48, -7896]), ([87.5, 5.3])
  )

  # For powerrat
  powerratInvalids = (
    ([0, 0, 5]), ([0, 0, 0]), ([-5, 0, 3]), ([-5, 0, 0]), 
    ([-48, -7896, 42]), ([87.5, 5.3, 9]), ([87.5, 5, 9.5]), ([87, 5.3, 9.2]),
    ([87.5, 5.3, 0]), ([87315, 53, 0])
  )

  numberString = ''

  # ====================================================================
  # Tests for powerint
  print('='*60, "\nTesting powerint()...\n", '-'*59)
  # Invalid combinations of x^p
  for base, exponent in powerintInvalids:
    numberString = f'({base})^({exponent})'
    print ("Testing: {0:<30}\t\t".format(numberString), end=' ')
    
    try:
      powerint(base, exponent)
      assert False, "X ERROR X"
    except RuntimeError:
      assert True
    except ArithmeticError:
      assert True
    except TypeError:
      assert True
    print("-> Pass <-")

  # Valid integer combinations of x^p
  for base, exponent in simpleIntPowers:
    numberString = f'({base})^({exponent})'
    print ("Testing: {0:<30}\t\t".format(numberString), end=' ')
    test_powerint(base, exponent)
    print("-> Pass <-")

  # Valid floating point combinations of x^p (p is always integer)
  for base, exponent in simpleFloatPowers:
    numberString = f'({base})^({int(exponent)})'
    print ("Testing: {0:<30}\t\t".format(numberString), end=' ')
    test_powerint(base, exponent, dataType=float)
    print("-> Pass <-")

  # ====================================================================
  # Tests for powerrat
  print('='*60, "\nTesting powerrat()...\n", '-'*59)
  
  # Invalid combinations of x^(p/q)
  for base, exponentN, exponentD in powerratInvalids:
    numberString = f'({base})^({exponentN}/{exponentD})'
    print ("Testing: {0:<30}\t\t".format(numberString), end=' ')
    try:
      powerrat(base, exponentN, exponentD)
      assert False, "X ERROR X"
    except RuntimeError:
      assert True
    except ArithmeticError:
      assert True
    except TypeError:
      assert True
    print("-> Pass <-")
  
  # Valid integer combinations of x^(p/q)
  for base, expNum, expDenom in rationalIntPowers:
    numberString = f'({base})^({expNum}/{expDenom})'
    print ("Testing: {0:<30}\t\t".format(numberString), end=' ')
    test_powerrat(base, expNum, expDenom)
    print("-> Pass <-")

  # Valid floating point combinations of x^(p/q) (p and q are always integers)
  for base, expNum, expDenom in rationalFloatPowers:
    numberString = f'({base})^({expNum}/{expDenom})'
    print ("Testing: {0:<30}\t\t".format(numberString), end=' ')
    test_powerrat(base, expNum, expDenom, dataType=float)
    print("-> Pass <-")


def test_powerint(base, exponent, dataType=int):
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

  if base == 0:
    pyPow = 0
  elif exponent < 0 and base != 0:
    pyPow = 1/(base**((-1)*exponent))   
  else:
    pyPow = base**exponent
  
  # For debuging purposes only
  #print('{0} || {1}'.format(ans, pyPow))

  if dataType == int and exponent > 0:
    assert ans == pyPow, "X ERROR X"
  else:
    assert np.allclose(ans, pyPow), "X ERROR X"

def test_powerrat(base, exN, exD, dataType=int):
  """Test case for powerrat function.

  Calls powerrat and compares its answer with python's exponentiation
  function.
  The method math.isclose is always used to compare both results as it
  can take into account the powerrat's tolerance.

  This function will stop the code execution when it finds an error.

  Returns:
    None

  Raises:
    Nothing
  """

  ans = powerrat(base, exN, exD)

  if base == 0:
    pyPow = 0
  elif (exN < 0 and exD > 0) or (exN > 0 and exD < 0) and base != 0:
    pyPow = 1/(base**((abs(exN)/abs(exD))))
    
  else:
    pyPow = base**(exN/exD)

  # For debuging purposes only
  #print('{0} || {1}'.format(ans, pyPow))

  assert math.isclose(ans, pyPow, abs_tol=_POWERRAT_TOLERANCE), "X ERROR X"
  

if __name__ == "__main__":
  print("Executed as stand-alone script. Running test function.\n")

  initialTime = time.time_ns()

  test()

  finalTime = time.time_ns()

  print("Tests ran in:", end=' ')
  print(f"{(finalTime-initialTime)*10**(-9)} s ({finalTime-initialTime} ns)")