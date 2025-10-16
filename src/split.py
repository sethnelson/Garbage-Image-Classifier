import os
import numpy as np
from sklearn.model_selection import train_test_split

processed_path = "../data/small_test_set/processed"

X = np.load(processed_path + "/X.npy")
y = np.load(processed_path + "/y.npy")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.50)

np.save(os.path.join(processed_path, "X_train.npy"), X_train)
np.save(os.path.join(processed_path, "y_train.npy"), y_train)
np.save(os.path.join(processed_path, "X_val.npy"), X_val)
np.save(os.path.join(processed_path, "y_val.npy"), y_val)
np.save(os.path.join(processed_path, "X_test.npy"), X_test)
np.save(os.path.join(processed_path, "y_test.npy"), y_test)



