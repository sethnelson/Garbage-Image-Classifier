# Michael V recomments ResNet-18 and MobileNetV3
# MobileNetV3 designed for embedded systems
# https://medium.com/@RobuRishabh/understanding-and-implementing-mobilenetv3-422bd0bdfb5a

# GEEKS FOR GEEKS EXAMPLE CODE

import torch.nn as nn
import torch


# copy of Dr. Santos' code
def eval_model(model, test_loader, critereon):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval() # evaluation mode

    # tracking vars
    correct = 0
    total = 0
    loss = 0
    iterations = 0

    with torch.no_grad():
        for features, labels in test_loader:
            # transfer tensors to device
            features = features.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = model(features)
            #loss
            loss += critereon(outputs, labels).item()
            iterations += 1
            # final prediction
            _, predicted = torch.max(outputs, 1)
            # add processed samples count to total tally
            total += labels.size(0)
            # compute number of correct samples
            correct += (predicted == labels).sum().item()

    return 100 * correct/total, loss/iterations

def weights_init_xavier(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

# copy of Dr. Santos' code
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, display=False):

    # initialize output arrays
    train_losses = []
    val_losses = []
    val_accs = []

    for epoch in range(num_epochs):
        running_loss = 0.0 #* running loss value to smooth cost tracking over batches
        for i, (features, labels) in enumerate(train_loader):
            # zero gradients
            optimizer.zero_grad()
            # enable learning
            model.train()
            # move tensors to the device
            features = features.to(device)
            labels = labels.to(device)
            model.to(device)
            # forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            # backward and optimize
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * features.size(0)  #* sum up batch losses
        # evaluate after each epoch
        epoch_train_loss = running_loss / len(train_loader.dataset) # * the mean loss so far
        val_acc, val_loss = eval_model(model, test_loader, criterion)
        if display:
            print ('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}, ACC: {:.2f}'
                    .format(epoch+1, num_epochs, epoch_train_loss, val_loss, val_acc))
        #* append losses and validation accuracy after each epoch
        #train_losses.append(loss.item())
        train_losses.append(epoch_train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    return model, train_losses, val_losses, val_accs
