import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import logistic_regressor as lr #custom model

split_data = "../data/small_test_set/processed"

X_train = np.load(split_data + "/X_train.npy")
y_train = np.load(split_data + "/y_train.npy")
# X_val = np.load(split_data + "/X_val.npy")
# y_val = np.load(split_data + "/y_val.npy")
X_test = np.load(split_data + "/X_test.npy")
y_test = np.load(split_data + "/y_test.npy")

# Radical dimensionality reduction from 196,608 features to n_components
pca = PCA(n_components=155, # 90% explainability after graphing explained variance ratio
          whiten=True)      # appreciably boosts performance
X_train_pca = pca.fit_transform(X_train)
X_test_pca  = pca.transform(X_test)

model = LogisticRegression(
    C=0.36,         # determined by running 5-fold CV
    penalty='l2',   # using ridge to combat overfitting, lasso unavailable for multiclass
    max_iter=600,   # some reasonable value
    solver="lbfgs", # default solver acceptable
    n_jobs=-1)

model.fit(X_train_pca, np.argmax(y_train, axis=1)) # train and evaluate PCA
print("PCA Train accuracy:", model.score(X_train_pca, np.argmax(y_train, axis=1)))
print("PCA Test accuracy:", model.score(X_test_pca, np.argmax(y_test, axis=1)))