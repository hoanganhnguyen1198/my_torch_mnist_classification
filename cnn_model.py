import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_stack = nn.Sequential(
            # [1, 28, 28]
            nn.Conv2d( 
                in_channels=1, 
                out_channels=3, 
                kernel_size=3
            ),
            # [3, 26, 26]
            nn.MaxPool2d(
                kernel_size=2, 
                stride=2
            ),
            # [3, 13, 13]
            nn.Conv2d(
                in_channels=3,
                out_channels=6,
                kernel_size=4
            ),
            # [6, 10, 10]
            nn.MaxPool2d(
                kernel_size=2, 
                stride=2
            ),
            # [6, 5, 5]
            nn.Flatten(),
            # [1, 6*5*5]
            nn.Linear(
                in_features=6*5*5, 
                out_features=120
            ),
            # [1, 120]
            nn.ReLU(),
            nn.Linear(
                in_features=120, 
                out_features=60
            ),
            # [1, 60]
            nn.ReLU(),
            nn.Linear(
                in_features=60, 
                out_features=10
            )
            # [1, 10]
        )
    
    def forward(self, x):
        x = self.layers_stack(x)
        return x