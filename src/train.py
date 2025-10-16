import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

split_data = "../data/small_test_set/processed"

X_train = np.load(split_data + "/X_train.npy")
y_train = np.load(split_data + "/y_train.npy")
X_val = np.load(split_data + "/X_val.npy")
y_val = np.load(split_data + "/y_val.npy")
X_test = np.load(split_data + "/X_test.npy")
y_test = np.load(split_data + "/y_test.npy")

# print(X_train.shape)
# print(y_train.shape)

model = LogisticRegression(penalty=None, max_iter=1000).fit(X_train, y_train.flatten())
