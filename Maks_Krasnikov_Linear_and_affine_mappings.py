import numpy as np

def negative_matrix(x: np.ndarray) -> np.ndarray:
    """
    Returns the negation of each element in the input vector or matrix.

    Args:
        x (np.ndarray): A vector (n*1) or matrix (n*n).

    Returns:
        np.ndarray: A matrix with each element negated.
    """
    return -x

def reverse_matrix(x: np.ndarray) -> np.ndarray:
    """
    Returns the input vector or matrix with the order of elements reversed.

    Args:
        x (np.ndarray): A vector (n*1) or matrix (n*n).

    Returns:
        np.ndarray: A matrix with the order of elements reversed.
    """
    return np.flip(x)

def affine_transform(
    x: np.ndarray, alpha_deg: float, scale: tuple[float, float], shear: tuple[float, float],
    translate: tuple[float, float],
) -> np.ndarray:
    """Compute affine transformation

    Args:
        x (np.ndarray): vector n*1 or matrix n*n.
        alpha_deg (float): rotation angle in deg.
        scale (tuple[float, float]): x, y scale factor.
        shear (tuple[float, float]): x, y shear factor.
        translate (tuple[float, float]): x, y translation factor.

    Returns:
        np.ndarray: transformed matrix.
    """
    # Convert rotation angle to radians
    alpha_rad = np.deg2rad(alpha_deg)

    # Rotation matrix
    rotation_matrix = np.array([
        [np.cos(alpha_rad), -np.sin(alpha_rad)],
        [np.sin(alpha_rad), np.cos(alpha_rad)]
    ])

    # Scaling matrix
    scale_matrix = np.array([
        [scale[0], 0],
        [0, scale[1]]
    ])

    # Shear matrix
    shear_matrix = np.array([
        [1, shear[0]],
        [shear[1], 1]
    ])

    # Combine transformations (rotation, scaling, and shear)
    transform_matrix = rotation_matrix @ scale_matrix @ shear_matrix

    # Apply transformation to input matrix
    transformed = np.dot(transform_matrix, x.T).T

    # Apply translation
    if len(x.shape) == 2:  # For 2D matrices
        transformed += np.array(translate)
    elif len(x.shape) == 1:  # For 1D vectors
        transformed += np.array(translate)[:len(transformed)]

    return transformed
import numpy as np

# Початкова матриця
matrix = np.array([
    [1, 2],
    [3, 4]
])

# 1. Інверсія елементів
neg_matrix = negative_matrix(matrix)
print("Negative Matrix:")
print(neg_matrix)

# 2. Реверс матриці
rev_matrix = reverse_matrix(matrix)
print("Reversed Matrix:")
print(rev_matrix)

# 3. Афінне перетворення
alpha = 45  # Кут у градусах
scale = (1.5, 2.0)  # Масштабування по X та Y
shear = (0.5, 0.3)  # Зсув по X та Y
translate = (10, 5)  # Трансляція по X та Y

affine_matrix = affine_transform(matrix, alpha, scale, shear, translate)
print("Affine Transformed Matrix:")
print(affine_matrix)
