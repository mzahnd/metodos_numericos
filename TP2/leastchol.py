import numpy as np

def cholesky(matrix):
  
  # check if matrix is Cholesky compatible
  
  if matrix.shape[0] != matrix.shape[1]:
    return
  else:
    size = matrix.shape[0]
  eigenvalues, _ = np.linalg.eig(matrix)
  for value in eigenvalues:
    if value < 0:
      return

  # Now let's get down to business
  # Hay que limpiar un toque el código de abajo

  lower = np.zeros(matrix.shape)


  for i in range(size):
    for j in range(i+1):
      if i==j:
        lower[i][j] = np.sqrt(matrix[i][i] - np.sum(lower[i][:i]**2))
      else:
        sumatoria = []
        for p in range(j-1):
          sumatoria.append(lower[i][p]*lower[j][p])
        sumatoria = sum(sumatoria)
        lower[i][j] = (matrix[i][j] - sumatoria)/ lower[j][j]

  
  upper = np.matrix.transpose(lower)

  return lower, upper
   
matrix = np.array([[6, 3, 4, 8], [3, 6, 5, 1], [4, 5, 10, 7], [8, 1, 7, 25]])

L, U = cholesky(matrix)

print(L)
print(U)
print(np.matmul(L,U))
