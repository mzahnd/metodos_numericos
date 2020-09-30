# -*- coding: utf-8 -*-
"""

Sources:
  Warning management:
  https://stackoverflow.com/a/15934081
"""

import math
import numpy as np
import warnings

def powerint (x, p): 
  '''x: base; p:potencia
    devuelve x^p'''

  if x < 0:
    raise RuntimeError("Please provide a non negative number for the base.")
  elif type(p) is (not int or not np.intc or not np.int_ or not np.int8 \
                or not np.int16 or not np.int32 or not np.int64):
    raise TypeError("Exponent must be an int.")
  elif (x == 0 and p == 0):
    raise ArithmeticError("Math error. Trying to perform: 0^0")

  if x == 0:
      return 0

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
          if negativeExponent == True and y != 0:
            p *= -1
          
          raise OverflowError(f"Overflow while performing: ({x})^({p}).")
          #print("Overflow!!")
          #pass
      
  if negativeExponent == True and y != 0:
    return float(1/y)
  else:
    return y


def bisection(point, q, y): 
  '''Esta función es la que se evalúa para despejar 
  por bisección point = y^(1/q)'''
  return powerint(point, q) - y

def powerrat (x, p, q):
  
  y = powerint(x, p)
  
  if q<0:
    y = 1/y
    q *= -1
  
  tol = 0.0001
  
  a = 0
  b = 10

  while  bisection(a,q,y)*bisection(b,q,y)>0:
    print(bisection(a,q,y),bisection(b,q,y))
    b+=1000

  for _ in range(1000):
    c = (a+b)/2
    if (b-a)>tol: 
      if bisection(a,q,y)*bisection(c,q,y)>0:
        a = c
      else:
        b = c
    else:
      #print(c, b, a)
      return c
  #print(c, b, a)
  return c

#print(powerrat (2,3, -4))

def test():
  """---"""

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

  print('='*60, "\nTesting powerint()...\n", '-'*60, '\n')
  for base, exponent in simpleIntPowers:
    print (f"Testing: ({base})^({exponent})\t\t", end=' ')
    test_powerint(base, exponent)
    print("-> Pass <-")

  for base, exponent in simpleFloatPowers:
    print (f"Testing: ({base})^({np.int64(exponent)})\t\t", end=' ')
    test_powerint(base, exponent, dataType=float)
    print("-> Pass <-")

  print('='*60, "\nTesting powerrat()...\n", '-'*60, '\n')
  for base, expNum, expDenom in rationalIntPowers:
    print (f"Testing: ({base})^({exponent})\t\t", end=' ')
    test_powerrat(powerrat, base, expNum, expDenom)
    print("-> Pass <-")

  for base, expNum, expDenom in rationalFloatPowers:
    print (f"Testing: ({base})^({int(exponent)})\t\t", end=' ')
    test_powerrat(powerrat, base, expNum, expDenom, dataType=float)
    print("-> Pass <-")

def test_pow(fnAns, libAns, exponentNum, dType=int):
  if dType == int and exponentNum > 0:
    assert fnAns == libAns, "X ERROR X"
  else:
    assert np.allclose(fnAns, libAns), "X ERROR X"

def test_powerint(base, exponent, dataType=int):
  ans = powerint(base, exponent)

  print('{0}'.format(ans))

  if base == 0:
    libPow = 0
  elif exponent < 0 and base != 0:
    libPow = 1/(base**((-1)*exponent))
    print(libPow)
    
  else:
    libPow = base**exponent
    print(libPow)

  test_pow(ans, libPow, exponent, dType=dataType)

def test_powerrat(base, exN, exD, dataType=int):
  ans = powerrat(base, exN, exD)

  print('{0}'.format(ans))

  if base == 0:
    libPow = 0
  elif (exN < 0 or exD < 0) and base != 0:
    libPow = 1/(base**((abs(exN)/abs(exD))))
    print(libPow)
    
  else:
    libPow = base**(exN/exD)
    print(libPow)

  test_pow(ans, libPow, exN/exD, dType=dataType)


if __name__ == "__main__":
  print("Executed as stand-alone script. Running test function.\n")
  test()