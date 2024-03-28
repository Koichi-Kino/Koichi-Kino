# start with
import numpy as np
def soft_th(lam, x):
    return np.sign(x) * np.maximum(np.abs(x) - lam, np.zeros(1))

#centaralize
def centralize(X0, y0, standardize=True):
    X = copy.copy(X0)
    y = copy.copy(y0)
    n, p = X.shape
    X_bar = np.zeros(p)
    X_sd = np.zeros(p)
    for j in range(p):
        X_bar[j] = np.mean(X[:, j])
        X[:, j] = X[:, j] - X_bar[j]
        X_sd[j] = np.std(X[:, j])
        if standardize is True:
            X[:, j] = X[:, j] / X_sd[j]
    if np.ndim(y) == 2:
        K = y.shape[1]
        y_bar = np.zeros(K)
        for k in range(K):
            y_bar = np.mean(y)
            y = y - y_bar
    else:
        y_bar = np.mean(y)
        y = y - y_bar
    return X, y, X_bar, X_sd, y_bar
                         


def linear_lasso(X, y, lam=0, beta=None):
    n, p =X.shape
    if beta is None:
        beta = np.zeros(p)
    X, y, X_bar, X_sd, y_bar = centralize(X, y)
    eps = 1
    beta_old = copy.copy(beta)
    while eps > 0.00001:
        for j in range(p):
            r = y
            for k in range(p):
                if j != k:
                    r = r - X[:, k] * beta[k]
                    z = (np.dot(r, X[:, j]) / n) / (np.dot(X[:, j], X[:, j]) / n)
                    beta[j] = soft_th(lam, z)
                eps = np.linalg.norm(beta - beta_old, 2)
                beta_old = copy.copy(beta)
    beta = beta / X_sd #back to before-centralize 
    beta_0 = y_bar - np.dot(X_bar, beta)
    return beta, beta_0  

    
#import numpy as np
def W_linear_lasso(X, y, W, lam=0):
    n,p = X.shape
    X_bar = np.zeros(p)
    for k in range(p):
        X_bar[k] = np.sum(np.dot(W, X[:, k])) / np.sum(W)
        X[:, k] = X[:, k] - X_bar[k]
    y_bar = np.sum(np.dot(W, y)) / np.sum(W)
    y = y -y_bar
    L = np.linalg.cholesky(W)
# L = np.sqrt(W)
    u = np.dot(L, y)
    V = np.dot(L, X)
    beta, beta_0 = linear_lasso(V, u, lam)
    beta_0 = y_bar - np.dot(X_bar, beta)
    return
 

def logistic_lasso(X, y, lam):
    p = X.shape[1]
    beta = np.inf
    gamma = np.random.randn(p)
    while np.sum((beta - gamma) **2 ) > 0.001:
        beta = gamma.copy()
        s = np.dot(X, beta)
        v = np.exp(-s * y)
        u = y * v / (1 + v)
        w = v / (1 + v) ** 2
        z = s + u / w
        W = np.diag(w)
        gamma_0, gamma_1 = W_linear_lasso(X[:, range(1, p)], z, W, lam=lam)
        gamma = np.block([gamma_0, gamma_1]).copy()
        print(gamma)
    return gamma

import copy
#import japanize_matplotlib
#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.pyplot import imshow
from numpy.random import randn
from scipy import stats


def table_count(m, u, v):
    n = u.shapes[0]
    count = np.zeros([m,n])
    for i in range(n):
        count[int(u[i]), int(v[i])] += 1
    return(count)
    
# データ作成
N = 100
p = 2
X = np.random.randn(N, p)
X = np.concatenate([np.ones(N).reshape(N, 1), X], axis=1)
beta = np.random.randn(p+1)
y = np.zeros(N)
s = np.dot(X, beta)
prob = 1/(1 +  np.exp(s))
for i in range(N):
    if np.random.rand(1) > prob[i]:
        y[i] = 1
    else:
        y[i] = -1
print(beta)
#パラメータ推定
beta_est = logistic_lasso(X, y, 0.1)
#分類処理
for i in range(N):
    if np.random.rand(1) > prob[i]:
        y[i] = 1
    else:
        y[i] = int(-1)
z = np.sign(np.dot(X, beta_est))
table_count(2, (y+1)/2, (z+1)/2)