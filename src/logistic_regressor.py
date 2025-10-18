import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# __ FUNCTIONS __

# P(y=k) = e^z[k] / sum[j](e^z[j])
# https://medium.com/@jshaik2452/multi-class-logistic-regression-a-friendly-guide-to-classifying-the-many-4a590c2e6c26
def softmax(z):
     exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
     return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def initialize_params(n_classes, m_features, seed=None):
    if seed is not None:
         np.random.seed(seed)
    w = 0.005 * np.random.randn(m_features, n_classes) #small initial values
    b = np.zeros((1, n_classes)) #no initial bias
    return w, b

def forward(X, w, b):
    z = np.dot(X, w) + b #dot product of features and weights plus bias value
    y_hat = softmax(z) #compute the new hypothesis/prediction
    return z, y_hat

def compute_cost(y, y_hat, lam=None, reg=None, W=None): #compare predicted to actual
    n = y.shape[0]
    epsilon = np.finfo(y_hat.dtype).eps
    reg_term = 0
    if reg == 1:
            reg_term = (lam / n) * np.sum(np.abs(W))
    elif reg == 2:
            reg_term = (lam / (2 * n)) * np.linalg.norm(W)**2

    # https://www.geeksforgeeks.org/machine-learning/what-is-cross-entropy-loss-function/#
    # cross-entropy for multi class regression
    return - (1/n) * np.sum(y * np.log(y_hat + epsilon)) + reg_term

def compute_gradients(X, y, y_hat, lam=None, reg=None, W=None): #update the weights for each piece of data
    n = X.shape[0] # get size n from nxm matrix
    err = y_hat - y #create array of the error values
    dw = (1/n) * np.dot(X.T, err) #mean of values crossed with error
    db = (1/n) * np.sum(err, axis=0, keepdims=True) # mean cost
    if reg == 1:
            dw += (lam / n) * np.sign(W)
    elif reg == 2:
            dw += (lam / n) * W

    return dw, db

def update_params(w, b, dw, db, lr):
    w = w - (lr * dw)
    b = b - (lr * db)
    return w, b

def train(X, y, lr, n_epochs, lam=None, reg=None, X_val=None, y_val=None):
    m = X.shape[1] # extract size of m from nxm matrix
    n = y.shape[1] #number of classes
    w, b = initialize_params(n, m, 0) # starting set of random weights
    # containers for cost histories
    val_costs = []
    train_costs = []

    for epoch in range(n_epochs):
        _, y_hat = forward(X, w, b)
        train_costs.append(compute_cost(y, y_hat, lam, reg, w)) # update train history
        dw, db = compute_gradients(X, y, y_hat, lam, reg, w)
        w, b = update_params(w, b, dw, db, lr)
        if X_val is not None and y_val is not None:
            val_costs.append(compute_cost(y_val, predict_proba(X_val, w, b))) # update validation history
    return w, b, train_costs, val_costs

def predict_proba(X, w, b):
    return softmax(np.dot(X, w) + b) # squash predictions down to 0-1 scale, interpret as probability

def finalize(y_hat):
    return np.argmax(y_hat, axis=1)