import shap
import numpy as np
import torch
import matplotlib.pyplot as plt

#Shap implementation
#Used: https://medium.com/@oveis/easy-guide-using-shap-algorithm-to-explain-cnn-classification-of-sar-images-mstar-database-8138657585c8

def analyze_shap(model, Xtrain, Xtest, numBackground = 100, numSamples = 10):

    #Sample background images
    background = Xtrain[np.random.choice(Xtrain.shape[0], numBackground, replace = False)]

    explainer = shap.DeepExplainer(model, background) # Initialize the deepexplainer
    testSamples = Xtest[:numSamples] #Get test samples
    shapValues = explainer.shap_values(testSamples) #Compute Shap
    shap.image_plot(shapValues, testSamples) #plot Shap
    plt.savefig("../data/shap_plot.png", dpi=300, bbox_inches='tight') #save plot

    return shapValues, explainer

