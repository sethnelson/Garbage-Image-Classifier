# pair programming between Seth Nelson and Braeden Watkins

import numpy as np
from sklearn.linear_model import LogisticRegression

split_data = "../data/processed"
pca_cache = "../data/pca_cache"

# load in completely preprocessed train and test sets
X_train = np.load(pca_cache + "/X_train_pca.npy")
y_train = np.load(split_data + "/y_train.npy")
X_test = np.load(pca_cache + "/X_test_pca.npy")
y_test = np.load(split_data + "/y_test.npy")

model = LogisticRegression(
    C=0.36,         # determined by running 5-fold CV
    penalty='l2',   # using ridge to combat overfitting, lasso unavailable for multiclass
    max_iter=600,   # some reasonable value
    solver="lbfgs", # default solver acceptable
    n_jobs=-1)

model.fit(X_train, np.argmax(y_train, axis=1)) # train and evaluate PCA
print("PCA Train accuracy:", model.score(X_train, np.argmax(y_train, axis=1)))
print("PCA Test accuracy:", model.score(X_test, np.argmax(y_test, axis=1)))
