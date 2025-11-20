import shap
import numpy as np
import torch
import matplotlib.pyplot as plt

def inverseTransform(imgTensor): #reverse normalization done on images

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    return imgTensor.cpu() * std + mean

def plotShap(explainer, images, labels, targetIndex, name): #Calculate and plot the shap values

    batch = torch.cat(images, dim=0)
    shapValues = explainer.shap_values(batch)

    imgNumpy = np.transpose(inverseTransform(batch).numpy(), (0, 2, 3, 1))
    
    heatmaps = []
    count = len(images)

    for i in range(count):

        idx = targetIndex[i]

        if isinstance(shapValues, list):

            rawExplanation = shapValues[idx][i]
        else:

            rawExplanation = shapValues[i, :, :, :, idx]

        # Transpose (C, H, W) -> (H, W, C) and sum channels
        explanationHWC = np.transpose(rawExplanation, (1, 2, 0))
        heatmapSum = explanationHWC.sum(axis=-1, keepdims=True)
        heatmaps.append(heatmapSum)

    if len(heatmaps) > 0:
        
        shapValuesInput = [np.array(heatmaps)]
        print(f"Displaying {name}")
        shap.image_plot(shapValuesInput, imgNumpy[:count], labels=np.array(labels[:count]), show=True)

def analyzeShap(model, trainLoader, testLoader, device, classNames):
    print("\n--- Starting SHAP Analysis ---")
    
    batchFeatures, _ = next(iter(trainLoader))
    e = shap.GradientExplainer(model, batchFeatures[:50].to(device))

    # Correct Examples
    correctImgs, correctLbls, correctIndex = [], [], []
    foundClasses = set()

    for features, labels in testLoader:

        if len(foundClasses) >= 6: break

        features, labels = features.to(device), labels.to(device)
        _, preds = torch.max(model(features), 1)
        
        for i in range(len(labels)):

            lbl, pred = labels[i].item(), preds[i].item()

            if lbl == pred and lbl not in foundClasses:

                foundClasses.add(lbl)
                correctImgs.append(features[i].unsqueeze(0))
                correctLbls.append(f"{classNames[lbl]}\n(Correct)")
                correctIndex.append(pred)

                if len(foundClasses) >= 6: break

    if len(correctImgs) > 0:

        plotShap(e, correctImgs[:3], correctLbls[:3], correctIndex[:3], "Correct Examples (Batch 1)")

    if len(correctImgs) > 3:

        plotShap(e, correctImgs[3:], correctLbls[3:], correctIndex[3:], "Correct Examples (Batch 2)")

    # Incorrect Examples
    errorImgs, errorLbls, errorIndex = [], [], []

    for features, labels in testLoader:
        
        if len(errorImgs) >= 3: break

        features, labels = features.to(device), labels.to(device)
        _, preds = torch.max(model(features), 1)

        for i in range(len(labels)):

            lbl, pred = labels[i].item(), preds[i].item()

            if lbl != pred:

                errorImgs.append(features[i].unsqueeze(0))
                errorLbls.append(f"True: {classNames[lbl]}\nPred: {classNames[pred]}")
                errorIndex.append(pred) 
                
                if len(errorImgs) >= 3: break

    plotShap(e, errorImgs, errorLbls, errorIndex, "Incorrect Examples")