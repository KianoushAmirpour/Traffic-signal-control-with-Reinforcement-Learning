import torch 
import torch.nn as nn
import torch.nn.functional as F
# TODO: adding dropout or BN

class DQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()
        
        # architecture  of the model
        num_filters_cnn = [input_channels, 32, 64, 64]
        kernel_size = 3
        
        # convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size)
        self.conv2 = nn.Conv2d(32, 64, kernel_size)
        self.conv3 = nn.Conv2d(64, 64, kernel_size)
        
        # fully Connected layers
        self.fc1 = nn.Linear(20736, 512)
        self.fc2 = nn.Linear(512, num_actions)
        
        # initializing the weights
        for m in self.modules(): 
            if isinstance(m,  nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                   
    def forward(self, state): # input' shape: bs, 4, 24, 24
        
        x = F.relu(self.conv1(state)) # output' shape: bs, 32, 22, 22
        x = F.relu(self.conv2(x)) # output' shape: bs, 64, 20, 20
        x = F.relu(self.conv3(x)) # output' shape: bs, 64, 18, 18
        x = torch.flatten(x, 1) # output' shape: bs, 1, 20736
        x = F.relu(self.fc1(x)) # output' shape: bs, 1, 512
        x = self.fc2(x) # output' shape: bs, 1, 4
        
        return x
    
if __name__ == "__main__":
    x = torch.randn((2, 4, 24, 24))
    dqn = DQN(4, 4)
    print(dqn(x).shape, dqn(x))
    # print(dqn(x), dqn(x).argmax(dim=1)[0].cpu().numpy())