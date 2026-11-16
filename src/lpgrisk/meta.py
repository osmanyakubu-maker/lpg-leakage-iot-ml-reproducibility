"""L2-regularised multinomial logistic regression used as the stacking meta-learner.

Five-class (multinomial) form of the logistic loss in Eq. 11:
    L(W) = -sum_i w_i log softmax(W^T x_i)_{y_i} + (lambda/2) ||W||^2
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax


class SoftmaxRegression:
    def __init__(self, n_classes: int, l2: float = 1.0):
        self.K, self.l2 = n_classes, l2
        self.W = None

    @staticmethod
    def _aug(X):
        return np.hstack([X, np.ones((len(X), 1))])

    def _loss_grad(self, wflat, Xa, Y, sw):
        W = wflat.reshape(Xa.shape[1], self.K)
        logp = log_softmax(Xa @ W, axis=1)
        loss = -(sw * (Y * logp).sum(1)).sum()
        G = Xa.T @ ((np.exp(logp) - Y) * sw[:, None])
        Wr = W.copy(); Wr[-1] = 0  # bias not penalised
        return loss + 0.5 * self.l2 * (Wr ** 2).sum(), (G + self.l2 * Wr).ravel()

    def fit(self, X, y, sample_weight=None):
        Xa = self._aug(X)
        Y = np.eye(self.K)[y]
        sw = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        sw = sw / sw.mean()
        w0 = np.zeros(Xa.shape[1] * self.K)
        res = minimize(self._loss_grad, w0, args=(Xa, Y, sw), jac=True, method="L-BFGS-B",
                       options={"maxiter": 500})
        self.W = res.x.reshape(Xa.shape[1], self.K)
        return self

    def predict_proba(self, X):
        return softmax(self._aug(X) @ self.W, axis=1)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    # ---- on-line pieces (Algorithm 1) -----------------------------------
    def grad(self, X, y, sample_weight=None):
        """Mean gradient of the (weighted) cross-entropy over a mini-batch."""
        Xa = self._aug(X)
        Y = np.eye(self.K)[y]
        sw = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        P = softmax(Xa @ self.W, axis=1)
        return Xa.T @ ((P - Y) * sw[:, None]) / sw.sum()

    def fisher_diag(self, X, y):
        """Diagonal empirical Fisher information: E[(d log p(y|x) / dW)^2]."""
        Xa = self._aug(X)
        Y = np.eye(self.K)[y]
        P = softmax(Xa @ self.W, axis=1)
        D = P - Y                                   # (n, K)
        return np.einsum("ni,nk->ik", Xa ** 2, D ** 2) / len(X)
