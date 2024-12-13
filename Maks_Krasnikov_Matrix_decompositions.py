import numpy as np

def lu_decomposition(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform LU decomposition of a matrix.

    Args:
        x (np.ndarray): The input matrix to decompose.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            The permutation matrix P, lower triangular matrix L, and upper triangular matrix U.
    """
    from scipy.linalg import lu
    P, L, U = lu(x)
    return P, L, U

def qr_decomposition(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform QR decomposition of a matrix.

    Args:
        x (np.ndarray): The input matrix to decompose.

    Returns:
        tuple[np.ndarray, np.ndarray]: The orthogonal matrix Q and upper triangular matrix R.
    """
    Q, R = np.linalg.qr(x)
    return Q, R

def determinant(x: np.ndarray) -> np.ndarray:
    """
    Calculate the determinant of a matrix.

    Args:
        x (np.ndarray): The input matrix.

    Returns:
        np.ndarray: The determinant of the matrix.
    """
    return np.linalg.det(x)

def eigen(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the eigenvalues and right eigenvectors of a matrix.

    Args:
        x (np.ndarray): The input matrix.

    Returns:
        tuple[np.ndarray, np.ndarray]: The eigenvalues and the right eigenvectors of the matrix.
    """
    eigenvalues, eigenvectors = np.linalg.eig(x)
    return eigenvalues, eigenvectors

def svd(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Singular Value Decomposition (SVD) of a matrix.

    Args:
        x (np.ndarray): The input matrix to decompose.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The matrices U, S, and V.
    """
    U, S, V = np.linalg.svd(x)
    return U, S, V
import numpy as np

# Матриця для прикладів
matrix = np.array([
    [4, 3],
    [6, 3]
])

# 1. LU-розклад
P, L, U = lu_decomposition(matrix)
print("LU Decomposition:")
print("P:", P)
print("L:", L)
print("U:", U)

# 2. QR-розклад
Q, R = qr_decomposition(matrix)
print("\nQR Decomposition:")
print("Q:", Q)
print("R:", R)

# 3. Визначник
det = determinant(matrix)
print("\nDeterminant:")
print(det)

# 4. Власні значення та вектори
eigenvalues, eigenvectors = eigen(matrix)
print("\nEigenvalues and Eigenvectors:")
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:", eigenvectors)

# 5. Сингулярний розклад
U, S, V = svd(matrix)
print("\nSVD:")
print("U:", U)
print("S:", S)
print("V:", V)
