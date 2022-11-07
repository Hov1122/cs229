import numpy as np
import util
import matplotlib.pyplot as plt

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=True)
    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    clf = PoissonRegression(step_size=2e-7)
    clf.fit(x_train, y_train)
    print(clf.theta)
    # Run on the validation set, and use np.savetxt to save outputs to pred_path

    def plot(y_label, y_pred, title):
        plt.plot(y_label, 'go', label='label')
        plt.plot(y_pred, 'rx', label='prediction')
        plt.suptitle(title, fontsize=12)
        plt.legend(loc='upper left')

    y_train_pred = clf.predict(x_train)
    plot(y_train, y_train_pred, 'Training Set')

    y_valid_pred = clf.predict(x_valid)
    plot(y_valid, y_valid_pred, 'Validation Set')

    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***

        self.theta = np.zeros(x.shape[1])

        while True:
            step = self.step_size / x.shape[0] * (x.T @ (y - np.exp(x.dot(self.theta)))) # alpha = size/m
            self.theta += step
            if np.linalg.norm(step, 1) < self.eps:
                break

        return self.theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***

        return np.exp(x.dot(self.theta))

        # *** END CODE HERE ***
