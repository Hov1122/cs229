import numpy as np
import util
from functools import reduce
from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***

    # Train GDA
    model = GDA()
    model.fit(x_train, y_train)

    # Plot data and decision boundary
    util.plot(x_train, y_train, model.theta, 'output/p01e_{}.png'.format(pred_path[-5]))

    # Save predictions
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)

    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')

    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***

        m, n = x.shape
        self.theta = np.zeros([n + 1])

        fi = np.count_nonzero(y == 0) / m

        mu0 = np.sum(x.T, where=[y == 0], axis=1) / (m - fi * m)
        mu1 = np.sum(x.T, where=[y == 1], axis=1) / (fi * m)
        sigma = ((x[y == 0] - mu0).T.dot(x[y == 0] - mu0) + (x[y == 1] - mu1).T.dot(x[y == 1] - mu1)) / m
        sigma_inverse = np.linalg.inv(sigma)

        theta = sigma_inverse.dot(mu1 - mu0)
        theta0 = ((mu0 + mu1).T.dot(sigma_inverse).dot((mu0 - mu1)) - np.log((1 - fi) / fi)) / 2

        self.theta = np.insert(theta, 0, theta0)

        return self.theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***

        return x @ self.theta >= 0

        # *** END CODE HERE
