from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler

import os

# Aaron: Added imports
import pdb
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics, manifold
os.environ['KMP_DUPLICATE_LIB_OK']='True'

'''
This code is adapted from two sources:
(i) The official PyTorch MNIST example (https://github.com/pytorch/examples/blob/master/mnist/main.py)
(ii) Starter code from Yisong Yue's CS 155 Course (http://www.yisongyue.com/courses/cs155/2020_winter/)
'''

class fcNet(nn.Module):
    '''
    Design your model with fully connected layers (convolutional layers are not
    allowed here). Initial model is designed to have a poor performance. These
    are the sample units you can try:
        Linear, Dropout, activation layers (ReLU, softmax)
    '''
    def __init__(self):
        # Define the units that you will use in your model
        # Note that this has nothing to do with the order in which operations
        # are applied - that is defined in the forward function below.
        super(fcNet, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=20)
        self.fc2 = nn.Linear(20, 10)
        self.dropout1 = nn.Dropout(p=0.5)

    def forward(self, x):
        # Define the sequence of operations your model will apply to an input x
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = F.relu(x)

        output = F.log_softmax(x, dim=1)
        return output


class ConvNet(nn.Module):
    '''
    Design your model with convolutional layers.
    '''
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1)
        self.conv2 = nn.Conv2d(8, 8, 3, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(200, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output

# Design your own CNN
class Net(nn.Module):
    '''
    Build the best MNIST classifier.
    '''
    def __init__(self):
        super(Net, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.15),
            
            nn.Conv2d(16, 16, kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.15),
            
            nn.Flatten(),
            nn.Linear(400, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        
    def forward(self, x):
        x = self.model(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(args, model, device, train_loader, optimizer, epoch):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    model.train()   # Set the model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # Clear the gradient
        output = model(data)                # Make predictions
        loss = F.nll_loss(output, target)   # Compute loss
        loss.backward()                     # Gradient computation
        optimizer.step()                    # Perform a single optimization step
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()    # Set the model to inference mode
    test_loss = 0
    correct = 0
    test_num = 0
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_num += len(data)

    test_loss /= test_num

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_num,
        100. * correct / test_num))

    # return the loss and accuracy %
    return test_loss, 100. * correct / test_num

# Visualize 9 of the learned kernels from the first layer of your network
# and present in 3x3 grid
def visualizeKernels(model):
    # Get weights of the first layer, extract the first 9 kernels
    firstLayer = model.model[0].weight
    numKernels = firstLayer.shape[0]
    
    assert numKernels >= 9
    
    fig, axes = plt.subplots(3, 3)
    for i in range(3):
        for j in range(3):
            h = axes[i, j].imshow(np.squeeze(firstLayer[3*i+j].detach().numpy()))
            fig.colorbar(h, ax=axes[i,j])
    plt.tight_layout()
    
    return fig, axes
    
def visualizeErrors(model, device, test_loader):
    model.eval()    # Set the model to inference mode
    
    incorrectInputs = []
    correctOutput = []
    predictedOutput = []
    incorrectFound = 0
    
    with torch.no_grad(): # For the inference step, gradient is not computed
        flag = False
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            wrongInds = np.squeeze(np.where(np.squeeze(pred.numpy()) != np.squeeze(target.view_as(pred).numpy())))
            
            for ind in wrongInds:
                correctOutput.append(target[ind].item())
                predictedOutput.append(pred[ind].item())
                incorrectInputs.append(data[ind])
                incorrectFound += 1
                if incorrectFound >= 9:
                    flag = True
                    break
            if flag:
                break
    
    fig, axes = plt.subplots(3, 3)
    for i, image in enumerate(incorrectInputs):
        im = np.squeeze(image.numpy())
        axes[i // 3, i % 3].imshow(im, cmap='gray')        
        axes[i // 3, i % 3].set_title(r'$Y_{pred} =$ ' + str(predictedOutput[i]) + r', $Y_{true} =$ ' + str(correctOutput[i]))
    plt.tight_layout()
    
    return fig, axes, incorrectInputs, correctOutput, predictedOutput
    
def clusteringVisualization(model, device, test_loader):
    model.eval() # Set the model to inference mode
    
    featureVectors = []
    labels = []
    samples = []
    dataList = []
    with torch.no_grad(): # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            x = data
            for i, layer in enumerate(model.model):
                if i == len(model.model)-1:
                    continue
                x = layer(x)
            featureVectors.extend(x.numpy().tolist())
            labels.extend(target.numpy().tolist())
            dataList.extend([data[i] for i in range(len(data))])
            
    featureVectors = np.array(featureVectors)
    labels = np.array(labels)
        
    embedded = manifold.TSNE(n_components=2).fit_transform(featureVectors)
    
    cm = plt.get_cmap('gist_rainbow')
        
    fig1, axes1 = plt.subplots()
    for i, val in enumerate(np.unique(labels)):
        points = embedded[np.where(labels == val)] 
        axes1.scatter(points[:,0], points[:,1], color=cm(i * 20), label=str(val))
    plt.legend(fontsize=10)
    plt.title('Visualizing processed inputs using tSNE')
    
    fig2, axes2 = plt.subplots(4,8)
    
    for i in range(4):
        x = featureVectors[i]
        closestInds = np.argsort(np.linalg.norm(x - featureVectors, axis=1))[1:9]
        for j, ind in enumerate(closestInds):
            axes2[i, j].imshow(np.squeeze(dataList[ind]), cmap='gray')
            axes2[i,j].axis('off')            
    fig2.suptitle('Inputs which give similar feature vectors')
    
    return fig1, axes1, fig2, axes2, embedded, featureVectors, labels

def confusionMatrix(model, device, test_loader):
    model.eval()    # Set the model to inference mode
    
    y_true = []
    y_pred = []
    
    with torch.no_grad(): # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            y_pred.extend(np.squeeze(pred.numpy()).tolist())
            y_true.extend(np.squeeze(target.numpy()).tolist())
            
    return metrics.confusion_matrix(y_true, y_pred, labels=[i for i in range(10)])
    
def loadModel(filename, device='cpu'):
    model = Net().to(device)
    model.load_state_dict(torch.load(filename))
    
    return model

def determineIndices(alpha=1, fracVal=0.15):
    # Load in MNIST without augmentation to create split
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([       # Data preprocessing
                    transforms.ToTensor(), 
                    transforms.Normalize((0.1307,), (0.3081,))
                    ]))
    
    subset_indices_train = set(range(len(train_dataset)))
    subset_indices_valid = set()
    
    labels = train_dataset.targets.numpy()
    groups = np.unique(labels)
    
    for group in groups:
        indices = np.where(labels == group)[0]
        # Select 15% to be in validation set
        subset_indices_valid = subset_indices_valid.union(set(np.random.choice(indices, round(fracVal * len(indices)), replace=False).tolist()))
        
    subset_indices_train = subset_indices_train.difference(subset_indices_valid)
        
    subset_indices_train = list(subset_indices_train)
    subset_indices_valid = list(subset_indices_valid)
    
    # For learning curve purposes, use alpha to control how much of train set
    # to use
    subset_indices_train = np.random.choice(subset_indices_train, round(alpha * len(subset_indices_train)), replace=False).tolist()
        
    return subset_indices_train, subset_indices_valid

# Create train, val, test data loaders optionally with data augmentation
# for train
def loadData(train_batch_size, test_batch_size, subset_indices_train, subset_indices_valid, augment=False, device='cpu'):
    # Experiment with data augmentation
    # Random vertical and horizontal flip is no good, can actually mess up the 
    # numbers into each other

    if augment:
        # Pytorch has default MNIST dataloader which loads data at each iteration
        train_dataset = datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.Compose([       # Data preprocessing
                        transforms.ToTensor(), 
                        # data augmentation
                        transforms.RandomRotation(45),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))
    else:
        # Pytorch has default MNIST dataloader which loads data at each iteration
        train_dataset = datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.Compose([       # Data preprocessing
                        transforms.ToTensor(), 
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size,
        sampler=SubsetRandomSampler(subset_indices_train)
    )
    
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=test_batch_size,
        sampler=SubsetRandomSampler(subset_indices_valid)
    )
    
    test_dataset = datasets.MNIST('../data', train=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=True)
        
    return train_loader, val_loader, test_loader

def main():
    # Training settings
    # Use the command line to modify the default settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 10)') # was 14
    parser.add_argument('--lr', type=float, default=5, metavar='LR',
                        help='learning rate (default: 5.0)')
    parser.add_argument('--step', type=int, default=1, metavar='N',
                        help='number of epochs between learning rate reductions (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate your model on the official test set')
    parser.add_argument('--load-model', type=str,
                        help='model file path')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    
    # Aaron added for generating learning curves
    parser.add_argument('--learning-curve', type=float, default=1/2, metavar='M', 
                        help='Fraction of train set to use')
    parser.add_argument('--augment', action='store_true', default=False, 
                        help='use data augmentation')
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Evaluate on the official test set
    # Should load in a saved model
    if args.evaluate:
        assert os.path.exists(args.load_model)
                
        temp = args.load_model.split('/')
        savePath = ''
        for i in range(len(temp)-1):
            savePath += temp[i]
            
        model = Net().to(device)
        model.load_state_dict(torch.load(args.load_model))

        test_dataset = datasets.MNIST('../data', train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        # Save the test loss for reusability
        np.save(savePath + '/testEval', test(model, device, test_loader))
              
        fig1, axes1, incorrectInputs, correctOutput, predictedOutput = visualizeErrors(model, device, test_loader)
        
        fig2, axes2 = visualizeKernels(model)
        
        confMat = confusionMatrix(model, device, test_loader)
        
        fig3, axes3, fig4, axes4, embedded, featureVectors, labels = clusteringVisualization(model, device, test_loader)
        
        print('Confusion Matrix: \n')
        print(confMat)
        
        np.save('confMat', confMat)
        fig1.savefig('visualizeErrors')
        fig2.savefig('visualizeKernels')
        fig3.savefig('tSNE')
        fig4.savefig('similarImages')
        
        return

    subset_indices_train, subset_indices_valid = determineIndices(args.learning_curve)

    train_loader, val_loader, _ = loadData(args.batch_size, args.test_batch_size, subset_indices_train, subset_indices_valid, args.augment)
    
    # Load your model [fcNet, ConvNet, Net]
    model = Net().to(device)

    # Try different optimzers here [Adam, SGD, RMSprop]
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    
    # Set your learning rate scheduler
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    trainEval = []
    valEval = []    
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        trainEval.append(test(model, device, train_loader))
        valEval.append(test(model, device, val_loader))
        scheduler.step()    # learning rate scheduler

        # You may optionally save your model at each epoch here
    # Each row is epoch, first column is loss, second column is % accuracy
    trainEval = np.array(trainEval)
    valEval = np.array(valEval)
    
    # Plot the train and validation loss across epoch
    fig1 = plt.figure()
    plt.title('Loss across Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(trainEval[:,0], label='Train')
    plt.plot(valEval[:,0], label='Validation')
    plt.legend()

    fig2 = plt.figure()
    plt.title('Accuracy across Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('% Accuracy')
    plt.plot(trainEval[:,1], label='Train')
    plt.plot(valEval[:,1], label='Validation')
    plt.legend()
    
    # If had data augmentation then also evaluate on original data
    if args.augment:
        origResults = []
        train_loader, val_loader, _ = loadData(args.batch_size, args.test_batch_size, subset_indices_train, subset_indices_valid, False)
        
        for loader in [train_loader, val_loader]:
            origResults.append(test(model, device, loader))
            
        origResults = np.array(origResults)
        
    if args.save_model:
        count = 0
        while os.path.isdir('model' + str(count)):
            count += 1
        path = 'model' + str(count)
        
        os.mkdir(path)
        
        torch.save(model.state_dict(), path + '/mnist_model.pt')
        
        # Also save the train and validation loss across epoch for reusability
        np.save(path + '/trainEval', np.array(trainEval))
        np.save(path + '/valEval', np.array(valEval))
        
        fp = open(path + '/README.txt', 'w')
        fp.write('Train (loss, acc): ' + str(trainEval[-1, 0]) + ' ' + str(trainEval[-1,1]) + '\n')
        fp.write('Val (loss, acc): ' + str(valEval[-1, 0]) + ' ' + str(valEval[-1,1]) + '\n')

        if args.augment:
            np.save(path + '/origResults', np.array(origResults))
            fp.write('Orig Train (loss, acc): ' + str(origResults[0, 0]) + ' ' + str(origResults[0,1]) + '\n')
            fp.write('Orig Val (loss, acc): ' + str(origResults[1, 0]) + ' ' + str(origResults[1,1]) + '\n')

        fig1.savefig(path + '/lossPlot')
        fig2.savefig(path + '/accPlot')
        
        fp.close()
        
    pdb.set_trace()
    
if __name__ == '__main__':
    main()
