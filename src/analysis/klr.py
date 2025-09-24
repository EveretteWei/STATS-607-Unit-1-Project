# --- Define Kernel Logistic Regression class ---
from scipy.special import expit
from scipy import optimize
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.optimize import _check_optimize_result
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import LabelBinarizer


def _loss_and_grad(w, K, y, alpha, clip=30):
    """Compute logistic-loss objective and its gradient under a kernel model.

    The linear predictor is `K @ w`, where `K` is a precomputed kernel matrix.
    We add an L2 penalty in the RKHS induced by `K`, scaled by `alpha = 1/C`.

    Objective:
        L(w) = sum_i [ -y_i * (K w)_i + log(1 + exp((K w)_i)) ] + (alpha/2) * w^T K w

    Args:
        w (np.ndarray): Coefficients in kernel space, shape (n_samples,).
        K (np.ndarray): Kernel Gram matrix, shape (n_samples, n_samples).
        y (np.ndarray): Binary targets in {0,1} (after LabelBinarizer), shape (n_samples,).
        alpha (float): L2 penalty factor, equal to 1 / C.
        clip (int): Unused placeholder kept for API compatibility.

    Returns:
        tuple[float, np.ndarray]: (loss, grad) where `loss` is a scalar float and
        `grad` has shape (n_samples,).
    """
    n_samples = K.shape[0]
    linear_prediction = K.dot(w)
    penalty = (alpha / 2.0) * w.T.dot(K).dot(w)
    out = np.sum(-y * linear_prediction + np.log(1 + np.exp(linear_prediction))) + penalty
    z = expit(linear_prediction)
    z0 = y - z - alpha * w
    grad = -K.dot(z0)
    return out, grad


def _kernel_logistic_regression_path(K, y, max_iter, tol=1e-4, coef=None, solver='lbfgs', C=1):
    """Optimize kernel logistic regression coefficients.

    Uses L-BFGS-B to minimize the objective in `_loss_and_grad`. If an initial
    coefficient vector is not provided, initializes `w0` as zeros.

    Args:
        K (np.ndarray): Kernel Gram matrix, shape (n_samples, n_samples).
        y (np.ndarray): Binary targets in {0,1}, shape (n_samples,).
        max_iter (int): Maximum optimizer iterations.
        tol (float): Gradient tolerance for convergence (passed to optimizer).
        coef (np.ndarray | None): Optional warm start vector for `w0`.
        solver (str): Currently only 'lbfgs' is supported.
        C (float): Inverse regularization strength (alpha = 1 / C).

    Returns:
        tuple[np.ndarray, int]: (coef, n_iter) where `coef` is the optimized
        coefficient vector with shape (n_samples,) and `n_iter` is the number
        of iterations reported by the optimizer.
    """
    n_samples = K.shape[0]
    func = _loss_and_grad
    if coef is None:
        w0 = np.zeros(n_samples, order='F', dtype=K.dtype)
    else:
        w0 = coef

    if solver == 'lbfgs':
        opt_res = optimize.minimize(
            func, w0, method="L-BFGS-B", jac=True,
            args=(K, y, 1.0 / C, 30),
            options={"gtol": tol, "maxiter": max_iter}
        )
    n_iter = _check_optimize_result(solver, opt_res, max_iter)
    w0, loss = opt_res.x, opt_res.fun
    return np.array(w0), n_iter


class KernelLogisticRegression(BaseEstimator, ClassifierMixin):
    """Kernel Logistic Regression estimator compatible with scikit-learn.

    This classifier fits a logistic model in the feature space induced by a
    positive semi-definite kernel. The decision function is linear in the
    RKHS: `f(x) = sum_i alpha_i * k(x, x_i)`.

    Parameters mirror common scikit-learn conventions and enable RBF or
    polynomial kernels by setting `kernel` and its hyperparameters.
    """

    def __init__(self, kernel='rbf', gamma=1, degree=3, coef0=1, C=1, tol=1e-4, max_iter=1000):
        """Initialize hyperparameters.

        Args:
            kernel (str): Kernel type passed to `pairwise_kernels` ('rbf', 'poly', ...).
            gamma (float): Kernel coefficient for RBF/poly.
            degree (int): Degree for polynomial kernel.
            coef0 (float): Independent term in poly kernels.
            C (float): Inverse regularization strength (larger C → less regularization).
            tol (float): Optimizer gradient tolerance.
            max_iter (int): Maximum L-BFGS-B iterations.
        """
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.C = C
        self.tol = tol
        self.max_iter = max_iter

    def _get_kernel(self, X, Y=None):
        """Compute the pairwise kernel matrix between X and Y.

        Args:
            X (np.ndarray): Array of shape (n_samples_X, n_features).
            Y (np.ndarray | None): Optional array (n_samples_Y, n_features). If None,
                computes the Gram matrix on X itself.

        Returns:
            np.ndarray: Kernel matrix with shape (n_samples_X, n_samples_Y) or
            (n_samples_X, n_samples_X) if Y is None.
        """
        params = {"gamma": self.gamma, "degree": self.degree, "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel, filter_params=True, **params)

    def fit(self, X, y):
        """Fit the classifier on training data.

        Performs input validation, binarizes labels to {0,1}, computes the
        kernel Gram matrix, and optimizes the coefficient vector.

        Args:
            X (np.ndarray): Training features of shape (n_samples, n_features).
            y (array-like): Training labels of shape (n_samples,).

        Returns:
            KernelLogisticRegression: Fitted estimator.
        """
        self.X_ = X
        X, y = check_X_y(X, y, accept_sparse=True)
        self.label_encoder_ = LabelBinarizer(neg_label=0, pos_label=1)
        y_ = self.label_encoder_.fit_transform(y).reshape((-1))
        self.classes_ = self.label_encoder_.classes_
        K = self._get_kernel(X)
        self.coef_, self.n_iter_ = _kernel_logistic_regression_path(
            K, y_, tol=self.tol, coef=None, C=self.C, solver='lbfgs', max_iter=self.max_iter
        )
        self.is_fitted_ = True
        return self

    def decision_function(self, X):
        """Compute raw decision scores f(x) for input samples.

        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features).

        Returns:
            np.ndarray: Scores with shape (n_samples,). Positive values favor class 1.
        """
        check_is_fitted(self, ["X_", "coef_"])
        K = self._get_kernel(X, self.X_)
        return K.dot(self.coef_)

    def predict(self, X):
        """Predict class labels for samples in X.

        Applies a zero threshold to the decision function and maps indices back
        to original class labels.

        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted labels with shape (n_samples,).
        """
        scores = self.decision_function(X)
        indices = (scores > 0).astype(int)
        return self.classes_[indices]

# --- End of Kernel Logistic Regression class ---
