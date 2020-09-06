#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trabajo Práctico: Cuadrados mínimos

@author: Batinic Rey, Joaquín
@author: Pompozzi, Magalí M.
@author: Zahnd, Martín E.
"""


import numpy as np


def cholesky(matrix):
    """Performs the cholesky descomposition into two matrices.

    Arguments:
        matrix: Hermitian and definite symmetric matrix to perfom Cholesky's
        descomposition.

    Raises:
        RuntimeError when an invalid matrix is given (ie. non Hermitian or
        non definite symmetric).

    Returns:
        Two numpy arrays.
    """
    # check if matrix is Cholesky compatible
    if matrix.shape[0] != matrix.shape[1]:
        raise RuntimeError("Matrix is not square.")

    # Is symmetric?
    if not np.allclose(matrix, np.transpose(matrix)):
        raise RuntimeError("Matrix is not symmetric.")
    else:
        size = matrix.shape[0]
    eigenvalues, _ = np.linalg.eig(matrix)
    for value in eigenvalues:
        if value < 0:
            raise RuntimeError("Matrix is not definite symmetric.")


    # Now the algorithm itself

    lower = np.zeros(matrix.shape)
    
    # We iterate over the triangle from left to right and top to bottom

    for i in range(size):
        for j in range(i + 1):  
            #if the element belongs to the diagonal:
            if i == j:
                lower[i][j] = np.sqrt(matrix[i][i] - np.sum(lower[i][:i]**2))
            # if the element doesn't belong to the diagonal
            else:
                sumatoria = []
                for z in range(j):
                    sumatoria.append(lower[i][z] * lower[j][z])
                sumatoria = sum(sumatoria)
                lower[i][j] = (matrix[i][j] - sumatoria) / lower[j][j]

    upper = np.matrix.transpose(lower)

    return lower, upper


def linealSolverForward(A, b):
    """Solves a system of equations given a lower triangular matrix.

    Arguments:
        A: Lower triangular matrix.
        b: System's solutions

    Raises:
        RuntimeError if A is not lower triangular

    Returns:
        A numpy array with the system's X vector.
    """
    #First we check if the matrix is a lower triangle
    if np.allclose(A, np.tril(A)):
        n = len(b)
        x = np.zeros((n, 1))

        #we apply the forward formula for every element of x 
        for k in range(0, n):
            tempSum = []
            for number in range(0, k):
                tempSum.append(-1 * A[k][number] * x[number])
            tempSum = sum(tempSum)
            x[k] = (b[k] + tempSum) / A[k][k]

        return x
        
    else:
        raise RuntimeError("Matrix A is not lower triangular.")


def linealSolverBackwards(A, b):
    """Solves a system of equations given a upper triangular matrix.

    Arguments:
        A: Lower triangular matrix.
        b: System's solutions

    Raises:
        RuntimeError if A is not upper triangular

    Returns:
        A numpy array with the system's X vector.
    """
    if np.allclose(A, np.triu(A)):
        n = len(b)
        x = np.zeros((n, 1))

        for k in reversed(range(0, n)):

            tempSum = []
            for number in range(k + 1, n):
                tempSum.append(-1 * A[k][number] * x[number])

            tempSum = sum(tempSum)
            x[k] = (b[k] + tempSum) / A[k][k]

        return x

    else:
        raise RuntimeError("Matrix A is not upper triangular.")

def leastsq(A, b):
    """Solves a least squares problem.

    Arguments:
        A: Numpy Matrix
        b: Numpy array with the points to approximate.

    Raises:
        RuntimeError if b is not a nx1 vector or the size of A (which is 
        square) is different than n, also if either A or b are not arrays.

    Returns:
        A numpy array with the system's approximation
    """

    if type(A) != np.ndarray or  type(b) != np.ndarray:
        raise RuntimeError("Input error! One of the leastq arguments is not a "
                            + "numpy array")

    if b.shape[1] != 1 or A.shape[0] != b.shape[0]:
        print('b', A.shape[1], b.shape[0])
        raise RuntimeError("b is not a nx1 vector or the size of A is "
            + "different than n.")

  # D is A^t*A and E is A^t*b
  # D x = E
    D = np.matmul(np.matrix.transpose(A), A)
    E = np.matmul(np.matrix.transpose(A), b)

  # Separates the lower and upper part of the cholesky decomposition of D
  # lowD uppD x = E
    lowD, uppD = cholesky(D)
    
  # W is equal to uppD x, and thus is the solution for LowD W = E  
    W = linealSolverForward(lowD, E)
    x = linealSolverBackwards(uppD, W)
    return x


def test():
    """Test cholesky, linealSolverForward, linealSolverBackwards and leastsq.

    leastsq depends on the other tree functions, so this ones are also 
    tested.
    
    This function will stop the code execution when it finds an error.

    Returns:
        None
    """

    # Definite positive matrices
    # This matrices eigenvalues are > 0
    positiveMatrices = (np.array([[2, 3], [3, 6]]), np.array([[1, 2], [2, 5]]),
                        np.array([[57, 40, 7], [40, 78, 6], [7, 6, 13]]),
                        np.array([[6, 3, 4, 8], [3, 6, 5, 1], [4, 5, 10, 7],
                                  [8, 1, 7, 25]]),
                        np.array([[34, 12, 0, 0], [12, 41, 0, 0], [0, 0, 1, 0],
                                  [0, 0, 0, 1]]))

    # Not definite positive matrices
    # This matrices eigenvalues are < 0
    notPositiveMatrices = (np.array([[4, 2],
                                     [2, -1]]), np.array([[-1, 1], [1, 1]]),
                           np.array([[3, 8, 9], [8, -5, 4], [9, 4, 0]]),
                           np.array([[57, 40, 77, 30], [40, 78, 61, 69],
                                     [77, 61, 13, 59], [30, 69, 59, 81]]))

    # Non Hermitian matrices.
    # As their values are reals, this is the same as 'non symmetric matrices'.
    notHermitian = (np.array([[1, 3], [9, 7]]),
                    np.array([[1, 5, 7], [2, 4, 9], [0, 3, 0]]),
                    np.array([[5, 7, 5, 2], [9, 2, 6, 2], [40, 54, 78, 84],
                              [10, 43, 21, 19]]))

    # ====== Cholesky ======
    print("="*60, "\nTesting cholesky() function...")
    # Definite positive matrices
    for testMatrix in positiveMatrices:
        print('-'*50)
        print(f'Testing definite positive matrix: \n{testMatrix}')
        try:
            A, B = cholesky(testMatrix)
            comparison = np.allclose(np.matmul(A, B), testMatrix)
            assert comparison, \
                "The last tested matrix did not pass the test."
            print("-> Pass <-")
        except RuntimeError:
            assert False, "The last tested matrix did not pass the test."

    # Not definite positive matrices
    # This matrices should raise and exception (these aren't valid)
    for testMatrix in notPositiveMatrices:
        print('-'*50)
        print(f'Testing not definite positive matrix: \n{testMatrix}')
        try:
            cholesky(testMatrix)
            assert False, "The last tested matrix did not pass the test."
        except RuntimeError:
            assert True
            print("-> Pass <-")

    # Not Hermitian matrices
    # This matrices should raise and exception (these aren't valid)
    for testMatrix in notHermitian:
        print('-'*50)
        print(f'Testing not Hermitian matrix: \n{testMatrix}')
        try:
            cholesky(testMatrix)
            assert False, "The last tested matrix did not pass the test."
        except RuntimeError:
            assert True
            print("-> Pass <-")

    # ====== Forward and backwards substitution ======
    # Each tuple has tree elements, by index:
    # 0: matrix A (lower triangular)
    # 1: matrix b (System solution)
    # 2: matrix c (Expected answer)
    lowTriangMatrices = (
        (np.array([[8, 0, 0], [2, 3, 0],
                   [4, 7, 1]]), np.array([[8], [5],
                                          [0]]), np.array([[1], [1], [-11]])),
        (np.array([[8, 0, 0], [2, 3, 0], [4, 7, 1]]), np.array([[5], [1],
                                                                [-8]]),
         np.array([[5 / 8], [-1 / 12], [-119 / 12]])),
        (np.array([[5, 0, 0], [76, 63, 0], [47, 77,
                                            31]]), np.array([[69], [10], [4]]),
         np.array([[69 / 5], [-742 / 45], [28127 / 1395]])),
        (np.array([[44, 0, 0, 0], [17, 10, 0, 0], [65, 43, 49, 0],
                   [75, 5, 81, 76]]), np.array([[66], [74], [8], [22]]),
         np.array([[3 / 2], [97 / 20], [-5961 / 980], [9747 / 1960]])),
    )

    upperTriangMatrices = (
        (np.array([[12, 5, 6], [0, 1, -4], [0, 0, 9]]),
         np.array([[-37], [2], [9]]), np.array([[-73 / 12], [6], [1]])),
        (np.array([[-8, 35, 65], [0, 40, -64],
                   [0, 0, 64]]), np.array([[20], [12], [43]]),
         np.array([[4595 / 512], [11 / 8], [43 / 64]])),
        
            (
            np.array([ [-72, 8, 64, 91], [0, 70, -90, -27], [0, 0, -39, -22], 
                       [0, 0, 0, -22] ]),
            np.array([ [-40], [18], [-47], [-2] ]),
            np.array([ [3499 / 1848], [3555 / 2002], [15 / 13], [1 / 11] ])
            )
         )

    print("="*60, "\nTesting linearSolverForward() function.")
    for system in lowTriangMatrices:
        try:
            print('-'*50)
            print("Testing system:\n"
                  + f"A =\n{system[0]}\n"
                  + f"b =\n{system[1]}\n"
                  + f"Expected answer =\n{system[2]}")

            ans = linealSolverForward(system[0], system[1])
            comparison = np.allclose(ans, system[2])
            assert \
                    comparison, \
                    "The last tested system did not pass the test."
            print("-> Pass <-")
        except RuntimeError:
            assert False, "The last tested system did not pass the test."

    print("="*60, "\nTesting linearSolverBackwards() function.")
    for system in upperTriangMatrices:
        try:
            print('-'*50)
            print("Testing system:\n"
                  + f"A =\n{system[0]}\n"
                  + f"b =\n{system[1]}\n"
                  + f"Expected answer =\n{system[2]}")

            ans = linealSolverBackwards(system[0], system[1])
            comparison = np.allclose(ans, system[2])
            assert \
                    comparison, \
                    "The last tested system did not pass the test."

            print("-> Pass <-")
        except RuntimeError:
            assert False, "The last tested system did not pass the test."

    # ====== leastsq() ======
    sqSystems = ( 
                (np.array([[1, 1], [4, 1], [9, 1]]),
                np.array([[3.1], [8.8], [20.2]]),
                np.array([[2109/980], [23/35]])),
                (np.array([[1, 1], [2, 1], [3, 1]]),
                np.array([[2.1], [3.9], [4.2]]),
                np.array([[21/20], [13/10]])),
                (np.array([[-10, 1], [-3, 1], [11, 1]]),
                np.array([[-3], [0], [9]]),
                np.array([[57/98], [117/49]]))
                )
                

    print("="*60, "\nTesting leastsq() function.")
    for system in sqSystems:
        try:
            print('-'*50)
            print("Testing system:\n"
                  + f"A =\n{system[0]}\n"
                  + f"b =\n{system[1]}\n"
                  + f"Expected answer =\n{system[2]}")
            ans = leastsq(system[0], system[1])
            comparison = np.allclose(ans, system[2])
            assert comparison, \
                "The last tested system did not pass the test."
            print("-> Pass <-")
        except RuntimeError:
            assert False, "The last tested system did not pass the test."


if __name__ == "__main__":
    print("Executed as stand-alone script. Running test function.\n")
    test()