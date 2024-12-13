from typing import Sequence
import numpy as np
from scipy import sparse

def get_vector(dim: int) -> np.ndarray:
    """Create random column vector with dimension dim.

    Args:
        dim (int): vector dimension.

    Returns:
        np.ndarray: column vector.
    """
    return np.random.rand(dim, 1)

def get_sparse_vector(dim: int) -> sparse.coo_matrix:
    """Create random sparse column vector with dimension dim.

    Args:
        dim (int): vector dimension.

    Returns:
        sparse.coo_matrix: sparse column vector.
    """
    data = np.random.rand(dim // 2)  # Half of the elements will be non-zero
    rows = np.random.choice(dim, size=dim // 2, replace=False)
    cols = np.zeros(dim // 2, dtype=int)  # Column index is always 0 for a column vector
    return sparse.coo_matrix((data, (rows, cols)), shape=(dim, 1))

def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vector addition.

    Args:
        x (np.ndarray): 1th vector.
        y (np.ndarray): 2nd vector.

    Returns:
        np.ndarray: vector sum.
    """
    return x + y

def scalar_multiplication(x: np.ndarray, a: float) -> np.ndarray:
    """Vector multiplication by scalar.

    Args:
        x (np.ndarray): vector.
        a (float): scalar.

    Returns:
        np.ndarray: multiplied vector.
    """
    return a * x

def linear_combination(vectors: Sequence[np.ndarray], coeffs: Sequence[float]) -> np.ndarray:
    """Linear combination of vectors.

    Args:
        vectors (Sequence[np.ndarray]): list of vectors of len N.
        coeffs (Sequence[float]): list of coefficients of len N.

    Returns:
        np.ndarray: linear combination of vectors.
    """
    return sum(c * v for c, v in zip(coeffs, vectors))

def dot_product(x: np.ndarray, y: np.ndarray) -> float:
    """Vectors dot product.

    Args:
        x (np.ndarray): 1th vector.
        y (np.ndarray): 2nd vector.

    Returns:
        float: dot product.
    """
    return float(np.dot(x.T, y))

def norm(x: np.ndarray, order: int | float) -> float:
    """Vector norm: Manhattan, Euclidean or Max.

    Args:
        x (np.ndarray): vector
        order (int | float): norm's order: 1, 2 or inf.

    Returns:
        float: vector norm
    """
    return float(np.linalg.norm(x, ord=order))

def distance(x: np.ndarray, y: np.ndarray) -> float:
    """L2 distance between vectors.

    Args:
        x (np.ndarray): 1th vector.
        y (np.ndarray): 2nd vector.

    Returns:
        float: distance.
    """
    return float(np.linalg.norm(x - y))

def cos_between_vectors(x: np.ndarray, y: np.ndarray) -> float:
    """Cosine between vectors in deg.

    Args:
        x (np.ndarray): 1th vector.
        y (np.ndarray): 2nd vector.

    Returns:
        np.ndarray: angle in deg.
    """
    cos_theta = dot_product(x, y) / (norm(x, 2) * norm(y, 2))
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

def is_orthogonal(x: np.ndarray, y: np.ndarray) -> bool:
    """Check is vectors orthogonal.

    Args:
        x (np.ndarray): 1th vector.
        y (np.ndarray): 2nd vector.

    Returns:
        bool: are vectors orthogonal.
    """
    return np.isclose(dot_product(x, y), 0.0)

def solves_linear_systems(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve system of linear equations.

    Args:
        a (np.ndarray): coefficient matrix.
        b (np.ndarray): ordinate values.

    Returns:
        np.ndarray: sytems solution
    """
    return np.linalg.solve(a, b)


def run_tests():
    # Генерація щільного та розрідженого вектора
    vec1 = get_vector(5)
    vec2 = get_vector(5)
    sparse_vec = get_sparse_vector(5)

    print("Vector 1:", vec1.flatten())
    print("Vector 2:", vec2.flatten())
    print("Sparse Vector:", sparse_vec.toarray().flatten())

    # Додавання векторів
    sum_vec = add(vec1, vec2)
    print("Sum of vec1 and vec2:", sum_vec.flatten())

    # Множення на скаляр
    scaled_vec = scalar_multiplication(vec1, 2.5)
    print("vec1 scaled by 2.5:", scaled_vec.flatten())

    # Лінійна комбінація
    combination = linear_combination([vec1, vec2], [0.5, -1.5])
    print("Linear combination of vec1 and vec2:", combination.flatten())

    # Скалярний добуток
    dot = dot_product(vec1, vec2)
    print("Dot product of vec1 and vec2:", dot)

    # Норми векторів
    l1_norm = norm(vec1, 1)
    l2_norm = norm(vec1, 2)
    max_norm = norm(vec1, np.inf)
    print("L1 norm of vec1:", l1_norm)
    print("L2 norm of vec1:", l2_norm)
    print("Max norm of vec1:", max_norm)

    # Відстань між векторами
    dist = distance(vec1, vec2)
    print("L2 distance between vec1 and vec2:", dist)

    # Косинус кута між векторами
    angle = cos_between_vectors(vec1, vec2)
    print("Cosine angle between vec1 and vec2 (deg):", angle)

    # Перевірка ортогональності
    orthogonal = is_orthogonal(vec1, vec2)
    print("Are vec1 and vec2 orthogonal?:", orthogonal)

    # Розв'язання системи лінійних рівнянь
    A = np.array([[3, 2], [1, 4]])
    b = np.array([6, 5])
    solution = solves_linear_systems(A, b)
    print("Solution of the linear system Ax = b:", solution)

# Запуск тестів
if __name__ == "__main__":
    run_tests()
