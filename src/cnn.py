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
            roc_auc_predictions = torch.softmax(outputs, 1)
            # add processed samples count to total tally
            total += labels.size(0)
            # compute number of correct samples
            correct += (predicted == labels).sum().item()

    return 100 * correct/total, loss/iterations, labels, predicted, roc_auc_predictions

# copy of Dr. Santos' code
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, display=False, smooth=True):

    # initialize output arrays
    train_losses = []
    val_losses = []
    val_accs = []

    for epoch in range(num_epochs):
        running_train_loss = 0.0 #* running loss value to smooth cost tracking over batches
        for i, (features, labels) in enumerate(train_loader):
            # zero gradients
            optimizer.zero_grad()
            # enable learning
            model.train()
            # move tensors to the device
            features = features.to(device)
            labels = labels.to(device)
            #model.to(device)
            # forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            # backward and optimize
            loss.backward()
            optimizer.step()
            if smooth:
              running_train_loss += loss.item() * features.size(0)  #* sum up batch losses
        # evaluate after each epoch
        val_acc, val_loss, _, _, _ = eval_model(model, val_loader, criterion)
        if smooth:
          epoch_train_loss = running_train_loss / len(train_loader.dataset) # * the mean loss so far
        if display:
            if smooth:
              print ('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}, ACC: {:.2f}'
                      .format(epoch+1, num_epochs, running_train_loss, val_loss, val_acc))
            else:
              print ('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}, ACC: {:.2f}'
                      .format(epoch+1, num_epochs, loss.item(), val_loss, val_acc))
        #* append losses and validation accuracy after each epoch
        if smooth:
          train_losses.append(epoch_train_loss)
        else:
          train_losses.append(loss.item())
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    return model, train_losses, val_losses, val_accs
