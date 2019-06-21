import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm


def MINE(x, y, n_epoch=200, representation='DV', hidden_parameters=10, 
         optimizer='Adam', batch_size=64, return_loss_plot=False, net="Additive", use_tqdm=True):
    
    class Additive_Net(nn.Module):  # possible choices for the net are possible
    # this has 5*H + 1 parameters
        def __init__(self, H=10):
            super(Additive_Net, self).__init__()
            self.fc1 = nn.Linear(1, H)
            self.fc2 = nn.Linear(1, H)
            self.fc3 = nn.Linear(H, 1)

        def forward(self, x, y):
            h1 = F.relu(self.fc1(x)+self.fc2(y))
            h2 = self.fc3(h1)
            return h2    
        
    class Concat_Net(nn.Module):
    # this has 4*H + 1 parameters
        def __init__(self, H=10):
            super(Concat_Net, self).__init__()
            self.fc1 = nn.Linear(2, H)
            self.fc3 = nn.Linear(H, 1)

        def forward(self, x, y):
            h1 = F.relu(self.fc1(torch.cat((x, y), 1)))
            h2 = self.fc3(h1)
            return h2  
        
        
    if batch_size is None: 
        batch_size = x.shape[0]  # use batch gradient descent~

    if net=="Additive":
        model = Additive_Net(hidden_parameters)    
    elif net=="Concat":
        model = Concat_Net(hidde)
    
    if (optimizer=='Adam'): 
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    elif (optimizer=='SGD'): 
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    plot_loss = []
    
    # use the standard scaler
    scaler = StandardScaler()        
    x = scaler.fit_transform(x)
    y = scaler.fit_transform(y)
    
    # convert to torch variable
    x_sample = Variable(torch.from_numpy(x).type(torch.FloatTensor), requires_grad = True)
    y_sample = Variable(torch.from_numpy(y).type(torch.FloatTensor), requires_grad = True)
    
    splits=np.arange(0, len(x)+batch_size, batch_size)  # find the indeces splitting the data in mini-batches
    
    if use_tqdm: 
        epoch_iterator = tqdm(range(n_epoch))
    else: 
        epoch_iterator = range(n_epoch)
        
    for epoch in epoch_iterator:
                
        y_shuffle=np.random.permutation(y)  # shuffle to be able to estimate independent average
        y_shuffle = Variable(torch.from_numpy(y_shuffle).type(torch.FloatTensor), requires_grad = True)    

        epoch_loss = 0
        
        for batch in range(len(splits)-1):

            x_batch = x_sample[splits[batch]:splits[batch+1]]
            y_batch = y_sample[splits[batch]:splits[batch+1]]
            y_shuffle_batch = y_shuffle[splits[batch]:splits[batch+1]]
            
            pred_xy = model(x_batch, y_batch)  # empirical average using joint prob.
            pred_x_y = model(x_batch, y_shuffle_batch)  # empirical average using shuffled (independent hypothesis) prob.

            if (representation=='DV'):
                ret = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
            elif (representation=='f'):
                ret = torch.mean(pred_xy) - torch.mean(torch.exp(pred_x_y) - 1)

            loss = - ret  # maximize
            epoch_loss += loss
            model.zero_grad()
            loss.backward()
            optimizer.step()
        
        plot_loss.append(epoch_loss.data.numpy()/(len(splits)-1))  # return the loss averaged over batches
        
    # to compute the average estimation I discard the first 25% of steps (arbitrary choice)
    if (return_loss_plot):
        return -np.mean(plot_loss[np.int(n_epoch/4):]), np.var(plot_loss[np.int(n_epoch/4):]), -np.array(plot_loss)
    else: 
        return -np.mean(plot_loss[np.int(n_epoch/4):]), np.var(plot_loss[np.int(n_epoch/4):])
        
def pairwise_neural_estimation_MI(df, n_epoch=200, representation='DV', hidden_parameters=10, 
                                  optimizer='Adam', batch_size=None):
    # each column of df represent a variable, each row an observation.
    # it returns 0 values in the case in which all rows are invalid for the given pair of variables
    
    num_vars = len(df.columns)
    
    corr_matrix = np.zeros((num_vars, num_vars, 2))
    
    for i in range(num_vars):
        for j in range(i+1, num_vars):
            pair = df.iloc[:,[i, j]].dropna()
            if len(pair) != 0:
                corr_matrix[i,j] = MINE(pair.iloc[:,0].values.reshape(-1, 1), 
                                    pair.iloc[:,1].values.reshape(-1, 1), n_epoch=n_epoch, 
                                    representation=representation, hidden_parameters=hidden_parameters, 
                                    optimizer=optimizer, batch_size=batch_size)
    
    return corr_matrix[np.triu_indices(num_vars, k=1)]