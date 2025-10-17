import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import logistic_regressor as lr #custom model

split_data = "../data/small_test_set/processed"

X_train = np.load(split_data + "/X_train.npy")
y_train = np.load(split_data + "/y_train.npy")
X_val = np.load(split_data + "/X_val.npy")
y_val = np.load(split_data + "/y_val.npy")
X_test = np.load(split_data + "/X_test.npy")
y_test = np.load(split_data + "/y_test.npy")

# print(X_train.shape)
# print(y_train.shape)

#model = LogisticRegression(penalty=None, max_iter=1000).fit(X_train, y_train.flatten())
#hyperparameters
learning_rate = 0.00001
number_epochs = 20
lam = None
reg = None

w, b, t_costs, v_costs = lr.train(X_train, y_train, learning_rate, number_epochs, lam, reg, X_val, y_val)

y_train_labels = np.argmax(y_train, axis=1)
y_val_labels   = np.argmax(y_val, axis=1)

y_pred_train = np.argmax(lr.predict_proba(X_train, w, b), axis=1)
y_pred_val   = np.argmax(lr.predict_proba(X_val, w, b), axis=1)

print("Training accuracy:", accuracy_score(y_train_labels, y_pred_train))
print("Validation accuracy:", accuracy_score(y_val_labels, y_pred_val))


plt.plot(t_costs, label='Training Loss')
plt.plot(v_costs, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Logistic Regression Training Progress')
plt.show()