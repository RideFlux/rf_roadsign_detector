import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, in_channels=1, num_classes=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

device = torch.device('cpu')
model = Classifier(in_channels=1, num_classes=9).to(device)
checkpoint = torch.load('classfier_checkpoints/best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
class ModelWithArgmax(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits = self.model(x)
        predicted_class = torch.argmax(logits, dim=1).to(torch.int32)
        return predicted_class

final_model = ModelWithArgmax(model)
final_model.eval()
dummy_input = torch.randn(1, 1, 17, 17, dtype=torch.float32)
onnx_file_path = "roadsign_classifier.onnx"
torch.onnx.export(
    final_model,
    dummy_input,
    onnx_file_path,
    input_names=['points'],
    output_names=['pred_class'],
)