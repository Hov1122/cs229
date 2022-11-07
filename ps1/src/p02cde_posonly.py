import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c

    # *** START CODE HERE ***

    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)

    # Train logistic regression
    model = LogisticRegression(eps=1e-5)
    model.fit(x_train, t_train)

    # Plot data and decision boundary
    x_eval, t_eval = util.load_dataset(test_path, label_col='t', add_intercept=True)
    util.plot(x_eval, t_eval, model.theta, './output/p02c.png')

    t_pred = model.predict(x_eval)
    np.savetxt(pred_path_c, t_pred, fmt='%d')

    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d

    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # Train logistic regression
    model = LogisticRegression(eps=1e-5)
    model.fit(x_train, y_train)

    # Plot data and decision boundary
    x_eval, y_eval = util.load_dataset(test_path, add_intercept=True)
    util.plot(x_eval, y_eval, model.theta, './output/p02d.png')

    y_pred = model.predict(x_eval)
    np.savetxt(pred_path_d, y_pred, fmt='%d')
    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e

    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)

    alpha = np.mean(1 / (1 + np.exp(-x_valid.dot(model.theta))), where=y_valid == 1)

    theta_prime = model.theta + np.log(2 / alpha - 1) * np.array([1, 0, 0])
    util.plot(x_valid, y_valid, theta_prime, './output/p02e.png')
    # *** END CODER HERE
