import torch.nn as nn

class WaldoRecognizer(nn.Module):
    def __init__(self, **kwargs):
        super(WaldoRecognizer, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.45),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Dropout(p=0.45),
            nn.Linear(256, 2)
        )

        # Apply Xavier initialization
        self._initialize_weights()

    def forward(self, x):
        return self.network(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)