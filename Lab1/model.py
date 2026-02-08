import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super(SimpleCNN, self).__init__()
        # Feature Extraction
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Input size: 128 channels * (IMG_SIZE/8) * (IMG_SIZE/8)
            # Với ảnh 224x224 -> qua 3 lần MaxPool (/8) -> 28x28
            nn.Linear(128 * 28 * 28, 512), 
            nn.ReLU(),
            nn.Dropout(p=dropout_rate), # Dropout cho Exp-2 [cite: 47]
            nn.Linear(512, 2) # Output: Cat, Dog
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def build_model(args):
    model = SimpleCNN(dropout_rate=args.dropout)
    return model