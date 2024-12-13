import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, load_diabetes

def preprocess(X: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
    """
    Preprocesses the input data by scaling features and splitting into training and test sets.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        list[np.ndarray]: List containing training and test sets for features and target.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

    return [X_train, X_test, y_train, y_test]

def get_regression_data() -> list[np.ndarray]:
    """
    Loads and preprocesses the diabetes dataset for regression tasks.

    Returns:
        list[np.ndarray]: List containing training and test sets for features and target.
    """
    data = load_diabetes()
    X, y = data.data, data.target
    return preprocess(X, y)

def get_classification_data() -> list[np.ndarray]:
    """
    Loads and preprocesses the breast cancer dataset for classification tasks.

    Returns:
        list[np.ndarray]: List containing training and test sets for features and target.
    """
    data = load_breast_cancer()
    X, y = data.data, data.target
    return preprocess(X, y)

def linear_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a linear regression model on the given data.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Trained linear regression model.
    """
    model = LinearRegression()
    model.fit(X, y)
    return model

def ridge_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a ridge regression model with hyperparameter tuning using GridSearchCV.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Best ridge regression model found by GridSearchCV.
    """
    param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
    model = Ridge()
    grid = GridSearchCV(model, param_grid, cv=5)
    grid.fit(X, y)
    return grid.best_estimator_

def lasso_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a lasso regression model with hyperparameter tuning using GridSearchCV.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Best lasso regression model found by GridSearchCV.
    """
    param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
    model = Lasso()
    grid = GridSearchCV(model, param_grid, cv=5)
    grid.fit(X, y)
    return grid.best_estimator_

def logistic_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a logistic regression model without regularization on the given data.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Trained logistic regression model.
    """
    model = LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000)
    model.fit(X, y)
    return model

def logistic_l2_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a logistic regression model with L2 regularization using GridSearchCV.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Best logistic regression model with L2 regularization found by GridSearchCV.
    """
    param_grid = {'C': [0.01, 0.1, 1.0, 10.0]}
    model = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000)
    grid = GridSearchCV(model, param_grid, cv=5)
    grid.fit(X, y)
    return grid.best_estimator_

def logistic_l1_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a logistic regression model with L1 regularization using GridSearchCV.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Best logistic regression model with L1 regularization found by GridSearchCV.
    """
    param_grid = {'C': [0.01, 0.1, 1.0, 10.0]}
    model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
    grid = GridSearchCV(model, param_grid, cv=5)
    grid.fit(X, y)
    return grid.best_estimator_
from sklearn.metrics import mean_squared_error, accuracy_score

# 1. Завантаження та підготовка даних
regression_data = get_regression_data()
classification_data = get_classification_data()

X_train_reg, X_test_reg, y_train_reg, y_test_reg = regression_data
X_train_cls, X_test_cls, y_train_cls, y_test_cls = classification_data

# 2. Регресія
# Лінійна регресія
lin_reg_model = linear_regression(X_train_reg, y_train_reg)
y_pred_lin_reg = lin_reg_model.predict(X_test_reg)
print("Linear Regression MSE:", mean_squared_error(y_test_reg, y_pred_lin_reg))

# Ridge-регресія
ridge_model = ridge_regression(X_train_reg, y_train_reg)
y_pred_ridge = ridge_model.predict(X_test_reg)
print("Ridge Regression MSE:", mean_squared_error(y_test_reg, y_pred_ridge))

# Lasso-регресія
lasso_model = lasso_regression(X_train_reg, y_train_reg)
y_pred_lasso = lasso_model.predict(X_test_reg)
print("Lasso Regression MSE:", mean_squared_error(y_test_reg, y_pred_lasso))

# 3. Класифікація
# Логістична регресія без регуляризації
def logistic_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a logistic regression model without regularization on the given data.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Trained logistic regression model.
    """
    model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
    model.fit(X, y)
    return model


# Логістична регресія з L2-регуляризацією
log_l2_model = logistic_l2_regression(X_train_cls, y_train_cls)
y_pred_log_l2 = log_l2_model.predict(X_test_cls)
print("Logistic Regression (L2) Accuracy:", accuracy_score(y_test_cls, y_pred_log_l2))

# Логістична регресія з L1-регуляризацією
log_l1_model = logistic_l1_regression(X_train_cls, y_train_cls)
y_pred_log_l1 = log_l1_model.predict(X_test_cls)
print("Logistic Regression (L1) Accuracy:", accuracy_score(y_test_cls, y_pred_log_l1))
# Висновки
# Після виконання завдання:
# 1) Порівняємо продуктивність моделей для регресії та класифікації:
#   * У випадку регресії очікується, що Ridge і Lasso будуть краще працювати на даних із високою мультиколінеарністю.
#   * У випадку класифікації регуляризація (L1 та L2) може допомогти уникнути перенавчання.
