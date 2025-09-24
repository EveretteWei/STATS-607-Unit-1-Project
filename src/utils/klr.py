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
    n_samples = K.shape[0]
    linear_prediction = K.dot(w)
    penalty = (alpha / 2.) * w.T.dot(K).dot(w)
    out = np.sum(-y * linear_prediction + np.log(1 + np.exp(linear_prediction))) + penalty
    z = expit(linear_prediction)
    z0 = y - z - alpha * w
    grad = -K.dot(z0)
    return out, grad

def _kernel_logistic_regression_path(K, y, max_iter, tol=1e-4, coef=None, solver='lbfgs', C=1):
    n_samples = K.shape[0]
    func = _loss_and_grad
    if coef is None:
        w0 = np.zeros(n_samples, order='F', dtype=K.dtype)
    else:
        w0 = coef
    if solver == 'lbfgs':
        opt_res = optimize.minimize(
            func, w0, method="L-BFGS-B", jac=True,
            args=(K, y, 1. / C, 30),
            options={"gtol": tol, "maxiter": max_iter}
        )
    n_iter = _check_optimize_result(solver, opt_res, max_iter)
    w0, loss = opt_res.x, opt_res.fun
    return np.array(w0), n_iter

class KernelLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, kernel='rbf', gamma=1, degree=3, coef0=1, C=1, tol=1e-4, max_iter=1000):
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
    
    def _get_kernel(self, X, Y=None):
        params = {"gamma": self.gamma, "degree": self.degree, "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel, filter_params=True, **params)

    def fit(self, X, y):
        self.X_ = X
        X, y = check_X_y(X, y, accept_sparse=True)
        self.label_encoder_ = LabelBinarizer(neg_label=0, pos_label=1)
        y_ = self.label_encoder_.fit_transform(y).reshape((-1))
        self.classes_ = self.label_encoder_.classes_
        K = self._get_kernel(X)
        self.coef_, self.n_iter_ = _kernel_logistic_regression_path(K, y_, tol=self.tol, coef=None, C=self.C, solver='lbfgs', max_iter=self.max_iter)
        self.is_fitted_ = True
        return self

    def decision_function(self, X):
        check_is_fitted(self, ["X_", "coef_"])
        K = self._get_kernel(X, self.X_)
        return K.dot(self.coef_)

    def predict(self, X):
        scores = self.decision_function(X)
        indices = (scores > 0).astype(int)
        return self.classes_[indices]
# --- End of Kernel Logistic Regression class ---