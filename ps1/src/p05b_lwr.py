import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def plot(x, y_label, y_pred, title):
    plt.figure()
    plt.plot(x[:,-1], y_label, 'bx', label='label')
    plt.plot(x[:,-1], y_pred, 'ro', label='prediction')
    plt.suptitle(title, fontsize=12)
    plt.legend(loc='upper left')
    plt.show()


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    lwlr = LocallyWeightedLinearRegression(tau=tau)
    # Fit a LWR model
    lwlr.fit(x_train, y_train)
    # Get MSE value on the validation set
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=True)
    y_valid_pred = lwlr.predict(x_valid)
    y_train_pred = lwlr.predict(x_train)
    # Plot validation predictions on top of training set
    y_train_pred = lwlr.predict(x_train)
    plot(x_train, y_train, y_train_pred, 'Training Set')

    y_valid_pred = lwlr.predict(x_valid)
    plot(x_valid, y_valid, y_valid_pred, 'Validation Set')
    # No need to save predictions
    # Plot data
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***

        self.x = x
        self.y = y

        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape

        w = np.exp(-np.linalg.norm((self.x - np.reshape(x, (m, 1, n))) ** 2, ord=2, axis=2) / (2 * self.tau ** 2))
        w = np.apply_along_axis(np.diag, axis=1, arr=w)

        theta = np.linalg.inv(self.x.T @ w @ self.x) @ self.x.T @ w @ self.y

        return (x * theta).sum(axis=1)
        # *** END CODE HERE ***
