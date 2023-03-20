import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score



if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


def run(model, train, test, config=None):

    #initialize variables
    # model = ConvModel(config['dropout']).to(device)
    criterion = nn.CrossEntropyLoss()


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

    
    #START
    for epoch in range(config['epoch']):

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
        #training variables end

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
        trainRunningLoss = trainRunningLoss/len(train) #TODO sjekk om dette faktisk er riktig???

        testRecall = recall_score(y_pred, y_true)
        testPrecision = precision_score(y_pred, y_true)

        bhAccuarcy = sum([x == y for (x, y) in zip(y_pred, y_true) if x == 0.0])/len([x for x in y_true if x == 0.0])
        sphAccuracy = sum([x == y for (x, y) in zip(y_pred, y_true) if x == 1.0])/len([x for x in y_true if x == 1.0])
        #test variables end

        #wandb log
        wandb.log({"Train epoch_loss":trainRunningLoss, 
                   "Test epoch_loss": testRunningLoss, 
                   "Train accuracy": trainAccuracy,
                   "Test accuracy": testAccuracy, 
                   "Test recall": testRecall, 
                   "Test precision": testPrecision,
                   "BH accuracy": bhAccuarcy,
                   "SPH accuracy": sphAccuracy})
        y_pred_out = y_pred
        y_true_out = y_true
        scheduler.step()

        #END
    return model, y_pred_out, y_true_out