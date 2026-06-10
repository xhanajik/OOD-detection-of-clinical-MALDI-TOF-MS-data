import torch
import torch.nn as nn
import math


def _init_hidden(m):
    """Initialization of hidden layers using He."""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def _init_output_layer(layer, num_classes):
    """
    Scale output layer to almost uniform weights 
    so that softmax doesnt collapse to one class.
    """
    std = 1.0 / math.sqrt(num_classes)
    nn.init.normal_(layer.weight, mean=0.0, std=std)
    nn.init.constant_(layer.bias, 0)


class TinyNN_NoDropout(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=128):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )
        self.output = nn.Linear(hidden_size, num_classes)

        self.hidden.apply(_init_hidden)
        _init_output_layer(self.output, num_classes)

    def forward(self, x):
        return self.output(self.hidden(x))


class TinyWideNN(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=512):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )
        self.output = nn.Linear(hidden_size, num_classes)

        self.hidden.apply(_init_hidden)
        _init_output_layer(self.output, num_classes)

    def forward(self, x):
        return self.output(self.hidden(x))


class TinyDeepNN(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=128, num_hidden_layers=3):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_size, num_classes)

        self.hidden.apply(_init_hidden)
        _init_output_layer(self.output, num_classes)

    def forward(self, x):
        return self.output(self.hidden(x))


class TinyNN_BatchNorm(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=128, num_hidden_layers=2):
        super().__init__()
        layers = [
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
        ]
        for _ in range(num_hidden_layers - 1):
            layers += [
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
            ]
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_size, num_classes)

        self.hidden.apply(_init_hidden)
        _init_output_layer(self.output, num_classes)

    def forward(self, x):
        return self.output(self.hidden(x))