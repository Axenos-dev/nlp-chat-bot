from torch import nn

# sipmle feed-forward model
class NLPModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        '''
        input_size  - in our case it will be the length of vocabulary
        hidden_size - can be everything, but i prefer to set it to 16
        output_size - in our case it will be amount of classes( tags )
        '''
        
        super().__init__()
        
        
        # I will use 3 linear layers connected with ReLU activation function. On output will be softmax activation
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        
        self.relu= nn.ReLU()
    
    # forward proccess
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        
        x = self.linear2(x)
        x = self.relu(x)
        
        output = self.output(x)
        
        return output