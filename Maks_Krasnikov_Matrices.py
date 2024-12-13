import numpy as np

def get_matrix(n: int, m: int) -> np.ndarray:
    """Create random matrix n * m.

    Args:
        n (int): number of rows.
        m (int): number of columns.

    Returns:
        np.ndarray: matrix n*m.
    """
    return np.random.rand(n, m)

def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Matrix addition.

    Args:
        x (np.ndarray): 1st matrix.
        y (np.ndarray): 2nd matrix.

    Returns:
        np.ndarray: matrix sum.
    """
    return np.add(x, y)

def scalar_multiplication(x: np.ndarray, a: float) -> np.ndarray:
    """Matrix multiplication by scalar.

    Args:
        x (np.ndarray): matrix.
        a (float): scalar.

    Returns:
        np.ndarray: multiplied matrix.
    """
    return x * a

def dot_product(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Matrices dot product.

    Args:
        x (np.ndarray): 1st matrix.
        y (np.ndarray): 2nd matrix or vector.

    Returns:
        np.ndarray: dot product.
    """
    return np.dot(x, y)

def identity_matrix(dim: int) -> np.ndarray:
    """Create identity matrix with dimension `dim`.

    Args:
        dim (int): matrix dimension.

    Returns:
        np.ndarray: identity matrix.
    """
    return np.eye(dim)

def matrix_inverse(x: np.ndarray) -> np.ndarray:
    """Compute inverse matrix.

    Args:
        x (np.ndarray): matrix.

    Returns:
        np.ndarray: inverse matrix.
    """
    return np.linalg.inv(x)

def matrix_transpose(x: np.ndarray) -> np.ndarray:
    """Compute transpose matrix.

    Args:
        x (np.ndarray): matrix.

    Returns:
        np.ndarray: transposed matrix.
    """
    return x.T

def hadamard_product(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute hadamard product.

    Args:
        x (np.ndarray): 1st matrix.
        y (np.ndarray): 2nd matrix.

    Returns:
        np.ndarray: hadamard product.
    """
    return np.multiply(x, y)

def basis(x: np.ndarray) -> tuple[int]:
    """Compute matrix basis.

    Args:
        x (np.ndarray): matrix.

    Returns:
        tuple[int]: indexes of basis columns.
    """
    _, _, pivot_columns = np.linalg.svd(x, full_matrices=False)
    return tuple(pivot_columns)

def norm(x: np.ndarray, order: int | float | str = 'fro') -> float:
    """Matrix norm: Frobenius, Spectral or Max.

    Args:
        x (np.ndarray): vector
        order (int | float | str): norm's order: 'fro', 2 or inf.

    Returns:
        float: vector norm
    """
    return np.linalg.norm(x, ord=order)
import numpy as np

# Приклад використання функцій
if __name__ == "__main__":
    # Створення матриць
    matrix1 = get_matrix(3, 3)  # Матриця 3x3
    matrix2 = get_matrix(3, 3)  # Інша матриця 3x3
    scalar = 2.5               # Скаляр

    print("Matrix 1:")
    print(matrix1)

    print("\nMatrix 2:")
    print(matrix2)

    # Додавання матриць
    sum_matrix = add(matrix1, matrix2)
    print("\nSum of Matrix 1 and Matrix 2:")
    print(sum_matrix)

    # Множення матриці на скаляр
    scaled_matrix = scalar_multiplication(matrix1, scalar)
    print("\nMatrix 1 multiplied by scalar (2.5):")
    print(scaled_matrix)

    # Добуток матриць (dot product)
    dot_result = dot_product(matrix1, matrix2)
    print("\nDot product of Matrix 1 and Matrix 2:")
    print(dot_result)

    # Одинична матриця
    identity = identity_matrix(3)
    print("\nIdentity Matrix (3x3):")
    print(identity)

    # Обернена матриця (для квадратної матриці)
    try:
        inverse_matrix = matrix_inverse(matrix1)
        print("\nInverse of Matrix 1:")
        print(inverse_matrix)
    except np.linalg.LinAlgError:
        print("\nMatrix 1 is singular and cannot be inverted.")

    # Транспонування матриці
    transpose_matrix = matrix_transpose(matrix1)
    print("\nTranspose of Matrix 1:")
    print(transpose_matrix)

    # Адамарів добуток
    hadamard = hadamard_product(matrix1, matrix2)
    print("\nHadamard product of Matrix 1 and Matrix 2:")
    print(hadamard)

    # Норма матриці (Фробеніусова)
    fro_norm = norm(matrix1, 'fro')
    print("\nFrobenius norm of Matrix 1:")
    print(fro_norm)

    # Норма матриці (максимальна за рядками)
    max_norm = norm(matrix1, np.inf)
    print("\nMax norm of Matrix 1:")
    print(max_norm)
