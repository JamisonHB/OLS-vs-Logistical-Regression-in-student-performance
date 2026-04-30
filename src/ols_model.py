import numpy as np
from metrics import rmse, r2_score
import matplotlib.pyplot as plt

class OrdinaryLeastSquares:
    def __init__(self):
        self.w = None
    
    def fit(self, X, y):
        self.w = np.linalg.pinv(X.T @ X) @ X.T @ y
        return self
    
    def predict(self, X):
        return X @ self.w
    
def add_bias(X):
    bias = np.ones((X.shape[0], 1))
    return np.hstack([bias, X])

def run_ols(X_train, y_train, X_test, y_test):
    X_train = add_bias(X_train)
    X_test = add_bias(X_test)

    model = OrdinaryLeastSquares()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("OLS RMSE:", rmse(y_test.values, y_pred))
    print("OLS R²:", r2_score(y_test.values, y_pred))

    # Parity plot
    plt.scatter(y_test, y_pred, alpha=.5)
    plt.plot([0, 20], [0, 20], 'r--')
    plt.title("OLS Predictions vs Actual")
    plt.show()