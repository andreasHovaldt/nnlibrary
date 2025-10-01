import torch.nn as nn
import torch.nn.functional as F



class SimpleMultiTaskModel(nn.Module):
    def __init__(self, input_dim, actuator_output_dim, mode_output_dim):
        super(SimpleMultiTaskModel, self).__init__()
        
        self.layer1 = nn.Linear(in_features=input_dim, out_features=128)
        self.layer2 = nn.Linear(in_features=128, out_features=128)
        self.regression_head = nn.Linear(in_features=128, out_features=actuator_output_dim)
        self.classification_head = nn.Linear(in_features=128, out_features=mode_output_dim)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x_actuators = F.relu(self.regression_head(x))
        x_mode = F.softmax(self.classification_head(x), dim=1)
        
        return x_actuators, x_mode
    

class SimpleClassificationModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleClassificationModel, self).__init__()
        
        self.layer1 = nn.Linear(in_features=input_dim, out_features=128)
        self.layer2 = nn.Linear(in_features=128, out_features=128)
        self.classification_head = nn.Linear(in_features=128, out_features=output_dim)
        
        self.relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.softmax(self.classification_head(x))

    
class HVACModeMLP(nn.Module):
    def __init__(self, window_size: int, feature_dim: int, n_classes: int):
        super().__init__()
        self.window_size = window_size
        self.feature_dim = feature_dim
        in_dim = window_size * feature_dim
        
        self.net = nn.Sequential(
            nn.Flatten(),  # Flattens all dims except batch
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, window, features)
        return self.net(x)