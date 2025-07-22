import torch

'''
1- ANN model consisting of 2 hidden layers and input and output layers
2- RELU activation function applied to hidden layers
3- Sigmoid applied to output layer (single neuron to produce a value between 0 and 1 resembling a probability)
'''

class FraudClassifier(torch.nn.Module):
    def __init__(self, x_dim):
        super(FraudClassifier, self).__init__()
        # linear layer 1 : input x 12
        self.linear1 = torch.nn.Linear(x_dim,12)
        # linear layer 2 : 12 x 6
        self.linear2 = torch.nn.Linear(12,6)
        # linear 3 : 6 x 1
        self.linear3 = torch.nn.Linear(6,1)
        # sigmoid and relu
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        # iniialize weights
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.linear1.bias.data.zero_()
        self.linear1.weight.data.uniform_(-initrange, initrange)
        self.linear2.bias.data.zero_()
        self.linear2.weight.data.uniform_(-initrange, initrange)
        self.linear3.bias.data.zero_()
        self.linear3.weight.data.uniform_(-initrange, initrange)

    def forward(self,x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        return self.sigmoid(self.linear3(x))
