# pair programming between Seth Nelson and Braeden Watkins

import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import shap

split_data = "../data/processed"
pca_cache = "../data/pca_cache"

# load in completely preprocessed train and test sets
X_train = np.load(pca_cache + "/X_train_pca.npy")
y_train = np.load(split_data + "/y_train.npy")
X_test = np.load(pca_cache + "/X_test_pca.npy")
y_test = np.load(split_data + "/y_test.npy")

# convert one-hot encoded labels back to single integers
# scitkit-learn expects class indices
y_train = np.argmax(y_train, axis=1)
y_test = np.argmax(y_test, axis=1)

model = LogisticRegression(
    C=0.36,         # determined by running 5-fold CV #.36
    penalty='l2',   # using ridge to combat overfitting, lasso unavailable for multiclass
    max_iter=600,   # some reasonable value #600
    solver="lbfgs", # default solver acceptable
    n_jobs=-1)

model.fit(X_train, y_train) # train and evaluate PCA

#these scores show how well the model did vs how well it performs on unseen data
train_acc = model.score(X_train, y_train) 
test_acc = model.score(X_test, y_test)

print(f"Train accuracy: {train_acc}")
print(f"Test accuracy:  {test_acc}")

#confusion matrix breaks down correctness, this breaks down what classes the model confuses with others
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# define class names in order
class_labels = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

#visulation using blue color scale(darker = more samples)
#diagonal line shows correct predictions, off diagonal are misclassifications
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap='Blues', xticks_rotation='vertical')

# resize the figure to make labels more readable
plt.gcf().set_size_inches(7, 6)

# adds a clear title and axis labels 
plt.title(f"Confusion Matrix - Logistic Regression\nTest Accuracy: {test_acc* 100:.1f}%")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout(pad=2.0)# needed for more expanded view, cant see the outside labels

# saving figure for the report

output_path = os.path.join("..", "data", "confusion_matrix.png") # builds file path
plt.savefig(output_path, dpi=300, bbox_inches='tight') # saves figure to file

# display it in a popup window
plt.show()

print(f"\n Confusion matrix saved to: {output_path}")

# shap analysis done by Joseph Kolly

explainer = shap.LinearExplainer(model, X_train[:100]) # create a shap explainer

shap_values = explainer.shap_values(X_test[:100]) # calculate the shap values for the test samples

shap.summary_plot(shap_values, X_test[:100]) # create a vizualization showing the results

plt.savefig("shapSummary.png") # save the plot as a png
