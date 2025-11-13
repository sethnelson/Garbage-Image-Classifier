# pair programming between Seth Nelson and Braeden Watkins

import os
import numpy as np
from sklearn.model_selection import train_test_split

processed_path = "../data/processed"

# load the prepossed data created by preprocess.py
X = np.load(processed_path + "/X.npy")
y = np.load(processed_path + "/y.npy")

# *(SN) switching to 80/20 split since we'll use Cross Validation instead of manual validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# save new sets for use by pca.py
np.save(os.path.join(processed_path, "X_train.npy"), X_train)
np.save(os.path.join(processed_path, "y_train.npy"), y_train)
np.save(os.path.join(processed_path, "X_test.npy"), X_test)
np.save(os.path.join(processed_path, "y_test.npy"), y_test)



