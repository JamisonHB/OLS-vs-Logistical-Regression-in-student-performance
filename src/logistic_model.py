import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from metrics import accuracy

class LogisticRegression:
    def __init__(self, lr=.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.loss_history = []
        self.val_loss_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def compute_loss(self, y, y_pred):
        epsilon = 1e-15
        return -np.mean(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))
    
    def fit(self, X, y, X_val=None, y_val=None):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.loss_history = []
        self.val_loss_history = []

        for _ in range(self.epochs):
            linear = X @ self.w
            y_pred = self.sigmoid(linear)

            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)

            gradient = (1 / n_samples) * (X.T @ (y_pred - y))
            self.w -= self.lr * gradient

            if X_val is not None:
                val_pred = self.sigmoid(X_val @ self.w)
                val_loss = self.compute_loss(y_val, val_pred)
                self.val_loss_history.append(val_loss)
    
    def predict_proba(self, X):
        return self.sigmoid(X @ self.w)
    
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
    
def add_bias(X):
    bias = np.ones((X.shape[0], 1))
    return np.hstack([bias, X])

def tune_hyperparameters(X_train, y_train, X_val, y_val):
    leaning_rates = [0.001, 0.01, 0.1]
    epochs_list = [500, 1000, 2000]

    best_acc = 0
    best_params = None

    for lr in leaning_rates:
        for epochs in epochs_list:
            model = LogisticRegression(lr=lr, epochs=epochs)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            acc = accuracy(y_val, y_pred)

            if acc > best_acc:
                best_acc = acc
                best_params = (lr, epochs)
    
    print("Best Params: ", best_params)
    return best_params

def plot_loss_curve(model):
    plt.plot(model.loss_history, label="Train Loss")
    if hasattr(model, "val_loss_history") and model.val_loss_history:
        plt.plot(model.val_loss_history, label="Validation Loss")
    plt.title("Learning Curve")
    plt.xlabel("Epoch")
    plt.ylabel("NLL Loss")
    plt.legend()
    plt.show()

def run_logistic(X_train, y_train, X_val, y_val, X_test, y_test):
    X_train = add_bias(X_train)
    X_val = add_bias(X_val)
    X_test = add_bias(X_test)

    best_lr, best_epochs = tune_hyperparameters(X_train, y_train, X_val, y_val)

    X_combined = np.vstack([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])

    model = LogisticRegression(lr = best_lr, epochs=best_epochs)
    model.fit(X_combined, y_combined, X_val, y_val)

    y_pred = model.predict(X_test)

    print("Accuracy: ", accuracy(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    plot_loss_curve(model)