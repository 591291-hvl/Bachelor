import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score

import lossFunctions
import nnmodels

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

def trainLoop(model, trainLoader, criterion, optimizer, train):
    #training variables
    trainRunningLoss = 0.0
    correct = 0
    y_pred = []
    y_true = []
    #training loop
    model.train()
    for i, data in enumerate(trainLoader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad() 
        output = model(inputs.permute(0,3,1,2)) # bytter på rekkefølge til dimensjonene 
        output1 = (torch.max(output.to(device), 1)[1]) # den predictede outputtn 
        y_pred.extend(output1) # Save Prediction
        
        y_true.extend(labels) # Save Truth
        loss = criterion(output, labels.type(torch.LongTensor).to(device))
        loss.backward()
        optimizer.step() # Endrer på vekten 
        
        trainRunningLoss += loss.item() * inputs.size(0) #
    #training loop end
    correct = (torch.FloatTensor(y_pred) == torch.FloatTensor(y_true)).sum()
    trainAccuracy = correct / len(y_true)
    trainRunningLoss = trainRunningLoss/len(train) #TODO sjekk om dette faktisk er riktig???
    #training variables end
    
    return trainAccuracy, trainRunningLoss


def testLoop(model, testLoader, criterion, test):
    #test variables
    testRunningLoss = 0.0
    y_pred = []
    y_true = []
    #test loop
    model.eval()
    for j, data in enumerate(testLoader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs.permute(0,3,1,2))# Feed Network

        output1 = (torch.max(output.to(device), 1)[1])
        y_pred.extend(output1) # Save Prediction
        
        y_true.extend(labels) # Save Truth
        loss = criterion(output, labels.type(torch.LongTensor).to(device))
        testRunningLoss += loss.item() * inputs.size(0)
    #test loop end

    correct = (torch.FloatTensor(y_pred) == torch.FloatTensor(y_true)).sum()
    testAccuracy = correct / len(y_true)
    testRunningLoss = testRunningLoss/len(test) #TODO sjekk om dette faktisk er riktig???

    y_pred_numpy = [x.data.cpu().numpy() for x in y_pred]
    y_true_numpy = [x.data.cpu().numpy() for x in y_true]

    testRecall = recall_score(y_pred_numpy, y_true_numpy)
    testPrecision = precision_score(y_pred_numpy, y_true_numpy)

    bhAccuarcy = sum([x == y for (x, y) in zip(y_pred_numpy, y_true_numpy) if x == 0.0])/len([x for x in y_true_numpy if x == 0.0])
    sphAccuracy = sum([x == y for (x, y) in zip(y_pred_numpy, y_true_numpy) if x == 1.0])/len([x for x in y_true_numpy if x == 1.0])
    #test variables end

    return testAccuracy, testRunningLoss, testRecall, testPrecision, bhAccuarcy, sphAccuracy, y_pred, y_true


def run(train, test, config=None):

    #initialize variables

    # ==== Models
    if config['model'] == 'cnn':
        model = nnmodels.ConvModel(config['dropout']).to(device)
    elif config['model'] == 'resnet34':
        model = nnmodels.ResNetModel(nnmodels.ResidualBlock, [3, 4, 6, 3]).to(device)
    elif config['model'] == 'resnet50':
        model = nnmodels.ResNet(nnmodels.Bottleneck, [3, 4, 6, 3], 2, 3).to(device)
    elif config['model'] == 'resnet101':
        model = nnmodels.ResNet(nnmodels.Bottleneck, [3, 4, 23, 3], 2, 3).to(device)
    
    # ==== Loss functions
    if config['loss'] == 'customLoss':
        criterion = lossFunctions.CustomLoss()
    elif config['loss'] == 'hinge':
        criterion = nn.MultiMarginLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    # ==== Optimizers
    if config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'])


    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['gamma'], last_epoch=-1)
    #initialize variables end

    #data
    trainLoader = DataLoader(train, shuffle=True, batch_size=config['batch_size'])
    testLoader = DataLoader(test, shuffle=True, batch_size=config['batch_size'])
    #data end


    wandb.watch(model, criterion, log='all')
    y_pred_out = []
    y_true_out = []

    # =========================================
    # pre test
    testAccuracy, testRunningLoss, testRecall, testPrecision, bhAccuarcy, sphAccuracy, y_pred, y_true = testLoop(model, testLoader, criterion, test)
    # =========================================
    wandb.log({ 
                   "Test epoch_loss": testRunningLoss, 
                   "Test accuracy": testAccuracy, 
                   "Test recall": testRecall, 
                   "Test precision": testPrecision,
                   "BH accuracy": bhAccuarcy,
                   "SPH accuracy": sphAccuracy}, step=0)

    #START
    for epoch in range(config['epoch']):
        # =========================================
        #Train
        trainAccuracy, trainRunningLoss = trainLoop(model, trainLoader, criterion, optimizer, train)
        # =========================================
        

        # =========================================
        #test
        testAccuracy, testRunningLoss, testRecall, testPrecision, bhAccuarcy, sphAccuracy, y_pred, y_true = testLoop(model, testLoader, criterion, test)
        # =========================================

        #wandb log
        wandb.log({"Train epoch_loss":trainRunningLoss, 
                   "Test epoch_loss": testRunningLoss, 
                   "Train accuracy": trainAccuracy,
                   "Test accuracy": testAccuracy, 
                   "Test recall": testRecall, 
                   "Test precision": testPrecision,
                   "BH accuracy": bhAccuarcy,
                   "SPH accuracy": sphAccuracy}, step = epoch +1)
        y_pred_out = y_pred
        y_true_out = y_true
        scheduler.step()

        #END
    return model, y_pred_out, y_true_out


def sweep(train, test, config=None):

    #initialize variables

    # ==== Models
    if config['model'] == 'cnn':
        model = nnmodels.ConvModel(config['dropout']).to(device)
    elif config['model'] == 'resnet34':
        model = nnmodels.ResNetModel(nnmodels.ResidualBlock, [3, 4, 6, 3]).to(device)
    elif config['model'] == 'resnet50':
        model = nnmodels.ResNet(nnmodels.Bottleneck, [3, 4, 6, 3], 2, 3).to(device)
    elif config['model'] == 'resnet101':
        model = nnmodels.ResNet(nnmodels.Bottleneck, [3, 4, 23, 3], 2, 3).to(device)
    
    # ==== Loss functions
    if config['loss'] == 'customLoss':
        criterion = lossFunctions.CustomLoss(config['exponent'])
    elif config['loss'] == 'hinge':
        criterion = nn.MultiMarginLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    # ==== Optimizers
    if config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'])


    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['gamma'], last_epoch=-1)
    #initialize variables end


    #data
    trainLoader = DataLoader(train, shuffle=True, batch_size=config.batch_size)
    testLoader = DataLoader(test, shuffle=True, batch_size=config.batch_size)
    #data end


    wandb.watch(model, criterion, log='all')

    
    # =========================================
    # pre test
    testAccuracy, testRunningLoss, testRecall, testPrecision, bhAccuarcy, sphAccuracy, y_pred, y_true = testLoop(model, testLoader, criterion, test)
    # =========================================
    wandb.log({ 
                   "Test epoch_loss": testRunningLoss, 
                   "Test accuracy": testAccuracy, 
                   "Test recall": testRecall, 
                   "Test precision": testPrecision,
                   "BH accuracy": bhAccuarcy,
                   "SPH accuracy": sphAccuracy}, step=0)

    #START
    for epoch in range(config['epoch']):
        # =========================================
        #Train
        trainAccuracy, trainRunningLoss = trainLoop(model, trainLoader, criterion, optimizer, train)
        # =========================================
        

        # =========================================
        #test
        testAccuracy, testRunningLoss, testRecall, testPrecision, bhAccuarcy, sphAccuracy, y_pred, y_true = testLoop(model, testLoader, criterion, test)
        # =========================================

        #wandb log
        wandb.log({"Train epoch_loss":trainRunningLoss, 
                   "Test epoch_loss": testRunningLoss, 
                   "Train accuracy": trainAccuracy,
                   "Test accuracy": testAccuracy, 
                   "Test recall": testRecall, 
                   "Test precision": testPrecision,
                   "BH accuracy": bhAccuarcy,
                   "SPH accuracy": sphAccuracy}, step = epoch +1)
        scheduler.step()

        #END
